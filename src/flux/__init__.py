try:
    from ._version import (
        version as __version__,  # type: ignore
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = Path(__file__).parent

# SAM3 + FLUX Inpainting exports
from .sam3_segmenter import SAM3Segmenter
from .inpainting import Flux2Inpainter, create_inpainter
from .sam_flux_pipeline import SAMFluxPipeline, InpaintResult, create_pipeline

__all__ = [
    "__version__",
    "version_tuple",
    "PACKAGE",
    "PACKAGE_ROOT",
    "SAM3Segmenter",
    "Flux2Inpainter",
    "create_inpainter",
    "SAMFluxPipeline",
    "InpaintResult",
    "create_pipeline",
]
