# Path: src/vlm_engineering/ingestion/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Document ingestion: PDF rendering, native text extraction, pluggable OCR."""

from .ocr import NullOCR, OCRAdapter
from .pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "OCRAdapter",
    "NullOCR",
]

try:  # pragma: no cover - optional dependency
    from .pdf import PDFAdapter

    __all__.append("PDFAdapter")
except ImportError:  # pragma: no cover - optional dependency
    PDFAdapter = None  # type: ignore[assignment]
