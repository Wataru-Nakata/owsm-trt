"""Fast OWSM (Open Whisper-style Speech Model) inference with TensorRT."""

from .pipeline import OWSM, Result, Segment

__version__ = "0.2.0"
__all__ = ["OWSM", "Result", "Segment", "__version__"]
