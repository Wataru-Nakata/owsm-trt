"""Bounded, inference-only frontend replay; returned tensors never alias captured outputs."""

import torch


class FrontendReplay:
    def __init__(self, frontend, max_entries: int = 32):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.frontend = frontend
        self.max_entries = max_entries
        self.entries = {}
        self.capture_stream = None
        self.ready = None
        self.fallbacks = 0

    def _entry(self, shape: tuple, dtype: torch.dtype, device: torch.device):
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if (
            device.type != "cuda"
            or shape[0] != 1
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
        ):
            return None
        key = (shape, dtype, device, self.frontend.fused_power, self.frontend.fused_logmel)
        if key not in self.entries:
            if len(self.entries) == self.max_entries:
                self.fallbacks += 1
                return None
            from .runtime import capture_graph

            if self.capture_stream is None:
                self.capture_stream = torch.cuda.Stream(device=device)
                self.ready = torch.cuda.Event()
            else:
                self.ready.synchronize()  # cold-only; prior workspace users are idle
            inp = torch.zeros(shape, dtype=dtype, device=device)
            out = None

            def body():
                nonlocal out
                out = self.frontend(inp)

            graph = capture_graph(body, self.capture_stream)
            self.entries[key] = (inp, out, graph)
        # Graphs on this capture stream share cuBLAS workspace and must never overlap.
        torch.cuda.current_stream(device).wait_event(self.ready)
        return self.entries[key]

    def _replay(self, entry) -> torch.Tensor:
        inp, out, graph = entry
        graph.replay()
        result = out.clone()  # an encoder on another stream may consume this after the next replay
        self.ready.record(torch.cuda.current_stream(inp.device))
        return result

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        entry = self._entry(tuple(speech.shape), speech.dtype, speech.device)
        if entry is None:
            return self.frontend(speech)
        entry[0].copy_(speech)
        return self._replay(entry)

    def from_wave(self, wave: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
        """Load one unpadded waveform directly into static [1,length] fp32 graph input."""
        entry = self._entry((1, length), torch.float32, device)
        inp = entry[0] if entry is not None else torch.empty((1, length), device=device)
        wave = wave[:length]
        if wave.device.type == "cpu" and inp.is_cuda:
            wave = wave.pin_memory()
        elif (
            wave.device == inp.device
            and wave.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
        ):
            wave = wave.clone()  # zeroing the padding must not erase a caller's aliased source
        # Every prefix element is overwritten by the copy; only padding needs initialization.
        if wave.shape[0] < length:
            inp[0, wave.shape[0] :].zero_()
        inp[0, : wave.shape[0]].copy_(wave, non_blocking=True)
        return self._replay(entry) if entry is not None else self.frontend(inp)

    def info(self) -> dict:
        return {
            "captures": len(self.entries),
            "max_entries": self.max_entries,
            "fallbacks": self.fallbacks,
            "input_output_bytes": sum(
                (inp.numel() * inp.element_size() + out.numel() * out.element_size())
                for inp, out, _ in self.entries.values()
            ),
            "workspace_bytes": None,
            "memory_note": "input/output only; captured intermediate private pools are additional",
        }
