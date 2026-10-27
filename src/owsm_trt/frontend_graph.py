"""Bounded, inference-only frontend replay; returned tensors never alias captured outputs."""

import weakref

import torch


def _release_capture(entries, ready, stream, destroy, handle):
    # Graph destruction can clear the stream's cuBLAS workspace. Keep that stream exclusive
    # until all its graphs and replay work are gone, including replays on encoder streams.
    try:
        ready.synchronize()
        stream.synchronize()  # also covers a failed warm-up/capture before ready was recorded
        entries.clear()
    finally:
        destroy(handle)


class FrontendReplay:
    def __init__(self, frontend, max_entries: int = 32, max_batch: int = 1, reuse_warmup_stream: bool = True):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_batch < 1:
            raise ValueError("max_batch must be positive")
        self.frontend = frontend
        self.reuse_warmup_stream = reuse_warmup_stream
        self.max_batch = max_batch
        self.max_entries = max_entries
        self.entries = {}
        self.capture_stream = None
        self.ready = None
        self.fallbacks = 0
        self.direct_replays = 0

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        if (
            speech.device.type != "cuda"
            or not 1 <= speech.shape[0] <= self.max_batch
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return self.frontend(speech)
        key = (
            tuple(speech.shape),
            speech.dtype,
            speech.device,
            self.frontend.fused_power,
            self.frontend.fused_logmel,
        )
        if key not in self.entries:
            if len(self.entries) == self.max_entries:
                self.fallbacks += 1
                return self.frontend(speech)
            # All captured frontend GEMMs use their own capture stream/workspace. They must not
            # overlap each other, or share the encoder/decoder's captured cuBLAS workspace.
            from .runtime import capture_graph

            if self.capture_stream is None:
                from cuda.bindings import runtime

                # torch.cuda.Stream() rotates through a finite shared pool. Another session's
                # graph destruction could then clear our eager warm-up's cuBLAS workspace
                # (PyTorch #193402). A native stream cannot alias that pool while it is alive.
                with torch.cuda.device(speech.device):
                    error, handle = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
                if error != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"frontend capture stream creation failed: {error}")
                try:
                    self.capture_stream = torch.cuda.ExternalStream(int(handle), device=speech.device)
                    self.ready = torch.cuda.Event()
                    self._release = weakref.finalize(
                        self, _release_capture, self.entries, self.ready, self.capture_stream,
                        runtime.cudaStreamDestroy, handle,
                    )
                except BaseException:
                    runtime.cudaStreamDestroy(handle)
                    raise
            else:
                self.ready.synchronize()  # cold-only; previous graph buffers/workspace are idle
            inp = torch.empty_like(speech)
            inp.copy_(speech)
            out = None

            def body():
                nonlocal out
                out = self.frontend(inp)

            graph = capture_graph(
                body, self.capture_stream,
                warmup_stream=self.capture_stream if self.reuse_warmup_stream else None,
            )
            self.entries[key] = (inp, out, graph)
        inp, out, graph = self.entries[key]
        stream = torch.cuda.current_stream(speech.device)
        stream.wait_event(self.ready)
        inp.copy_(speech)
        graph.replay()
        result = out.clone()  # the consumer may run after another stream reuses this graph
        self.ready.record(stream)
        return result

    def try_waves(self, waves: list[torch.Tensor], length: int, device: torch.device) -> torch.Tensor | None:
        """Fill an existing graph input directly; cold/unsupported calls use the normal capture path."""
        if (not self.entries or device.type != "cuda" or not 1 <= len(waves) <= self.max_batch
                or torch.is_grad_enabled() or torch.is_autocast_enabled("cuda")):
            return None
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = ((len(waves), length), torch.get_default_dtype(), device,
               self.frontend.fused_power, self.frontend.fused_logmel)
        entry = self.entries.get(key)
        if entry is None:
            return None
        inp, out, graph = entry
        stream = torch.cuda.current_stream(device)
        stream.wait_event(self.ready)
        try:
            inp.zero_()
            for row, wave in zip(inp, waves):
                wave = wave[:length]
                if wave.device.type == "cpu":
                    wave = wave.pin_memory()
                row[:wave.shape[0]].copy_(wave, non_blocking=True)
            graph.replay()
            result = out.clone()  # Consumers on another stream must not alias the next replay.
        finally:
            # A pin/copy failure may follow queued input writes; another stream must wait for those too.
            self.ready.record(stream)
        self.direct_replays += 1
        return result

    def info(self) -> dict:
        return {
            "captures": len(self.entries),
            "max_entries": self.max_entries,
            "max_batch": self.max_batch,
            "reuse_warmup_stream": self.reuse_warmup_stream,
            "capture_stream": "exclusive native stream",
            "fallbacks": self.fallbacks,
            "direct_replays": self.direct_replays,
            "input_output_bytes": sum(
                (inp.numel() * inp.element_size() + out.numel() * out.element_size())
                for inp, out, _ in self.entries.values()
            ),
            "workspace_bytes": None,
            "memory_note": "input/output only; captured intermediate private pools are additional",
        }
