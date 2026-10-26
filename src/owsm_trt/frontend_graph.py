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

    def __call__(self, speech: torch.Tensor) -> torch.Tensor:
        if speech.device.type != "cuda" or speech.shape[0] != 1 or torch.is_grad_enabled():
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
                self.capture_stream = torch.cuda.Stream(device=speech.device)
                self.ready = torch.cuda.Event()
            else:
                self.ready.synchronize()  # cold-only; previous graph buffers/workspace are idle
            inp = torch.empty_like(speech)
            inp.copy_(speech)
            out = None

            def body():
                nonlocal out
                out = self.frontend(inp)

            graph = capture_graph(body, self.capture_stream)
            self.entries[key] = (inp, out, graph)
        inp, out, graph = self.entries[key]
        stream = torch.cuda.current_stream(speech.device)
        stream.wait_event(self.ready)
        inp.copy_(speech)
        graph.replay()
        result = out.clone()  # the consumer may run after another stream reuses this graph
        self.ready.record(stream)
        return result

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
