import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing PDF libraries
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentParser:
    """
    Multi-format document parser (PDF, DOCX, MD, TXT).
    Applies text-first extraction for PDFs and falls back to pytesseract OCR
    only on pages with no extractable text layer.
    """

    @staticmethod
    def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
        results = []
        file_name = file_path.name

        # Method 1: Try pypdf / pdfplumber for native text
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for idx, page in enumerate(pdf.pages, start=1):
                        extracted_text = page.extract_text() or ""
                        
                        # Check requirement 5: Text-first layer check. Run OCR only if text < 20 chars
                        if len(extracted_text.strip()) < 20 and PYTESSERACT_AVAILABLE:
                            try:
                                page_img = page.to_image(resolution=200).original
                                ocr_text = pytesseract.image_to_string(page_img)
                                if len(ocr_text.strip()) > len(extracted_text.strip()):
                                    extracted_text = ocr_text.strip()
                                    logger.info(f"[{file_name}] Page {idx}: OCR fallback applied successfully.")
                            except Exception as ocr_err:
                                logger.debug(f"[{file_name}] Page {idx}: OCR fallback failed or Tesseract binary missing: {ocr_err}")

                        if extracted_text.strip():
                            results.append({
                                "text": extracted_text.strip(),
                                "page_number": idx,
                                "file_name": file_name,
                                "file_path": str(file_path),
                                "file_type": ".pdf"
                            })
                return results
            except Exception as e:
                logger.warning(f"[{file_name}] pdfplumber parsing failed: {e}. Falling back to pypdf...")

        if PYPDF_AVAILABLE:
            try:
                reader = pypdf.PdfReader(file_path)
                for idx, page in enumerate(reader.pages, start=1):
                    extracted_text = page.extract_text() or ""
                    if extracted_text.strip():
                        results.append({
                            "text": extracted_text.strip(),
                            "page_number": idx,
                            "file_name": file_name,
                            "file_path": str(file_path),
                            "file_type": ".pdf"
                        })
                return results
            except Exception as e:
                logger.error(f"[{file_name}] pypdf parsing error: {e}")

        return results

    @staticmethod
    def parse_docx(file_path: Path) -> List[Dict[str, Any]]:
        file_name = file_path.name
        if not DOCX_AVAILABLE:
            logger.error(f"[{file_name}] python-docx library is not available.")
            return []

        try:
            doc = docx.Document(file_path)
            full_text = []
            for p in doc.paragraphs:
                if p.text.strip():
                    full_text.append(p.text.strip())
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        full_text.append(row_text)

            combined_text = "\n\n".join(full_text)
            if not combined_text:
                return []

            return [{
                "text": combined_text,
                "page_number": 1,
                "file_name": file_name,
                "file_path": str(file_path),
                "file_type": ".docx"
            }]
        except Exception as e:
            logger.error(f"[{file_name}] Error parsing DOCX: {e}")
            return []

    @staticmethod
    def parse_text(file_path: Path) -> List[Dict[str, Any]]:
        file_name = file_path.name
        file_type = file_path.suffix.lower()

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.error(f"[{file_name}] Error reading text file: {e}")
                return []

        if not content.strip():
            return []

        return [{
            "text": content.strip(),
            "page_number": 1,
            "file_name": file_name,
            "file_path": str(file_path),
            "file_type": file_type
        }]

    @classmethod
    def parse_file(cls, file_path: Path) -> List[Dict[str, Any]]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return cls.parse_pdf(file_path)
        elif suffix == ".docx":
            return cls.parse_docx(file_path)
        elif suffix in (".md", ".txt"):
            return cls.parse_text(file_path)
        else:
            logger.debug(f"Skipping unsupported file format: {file_path.name}")
            return []
