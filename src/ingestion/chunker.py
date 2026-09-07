import re
from typing import List, Dict, Any
import config

class Chunker:
    """
    Metadata-aware chunking engine for parsed document pages.
    Splits text into overlapping chunks while preserving provenance (document name, page number).
    """

    def __init__(
        self,
        approx_token_size: int = config.CHUNK_SIZE_TOKENS,
        approx_token_overlap: int = config.CHUNK_OVERLAP_TOKENS
    ):
        self.chunk_size_chars = approx_token_size * config.APPROX_CHARS_PER_TOKEN  # ~2400 chars
        self.overlap_chars = approx_token_overlap * config.APPROX_CHARS_PER_TOKEN     # ~600 chars

    def create_chunks(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []

        for page in pages:
            file_name = page["file_name"]
            page_num = page.get("page_number", 1)
            raw_text = page["text"]
            file_path = page.get("file_path", "")
            file_type = page.get("file_type", "")

            # Split raw_text into paragraphs first
            paragraphs = re.split(r'\n\s*\n', raw_text)
            
            current_chunk_parts = []
            current_length = 0
            chunk_counter = 1

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # If paragraph itself exceeds max chunk size, break it up
                if len(para) > self.chunk_size_chars:
                    # Flush current chunk if non-empty
                    if current_chunk_parts:
                        chunk_text = "\n\n".join(current_chunk_parts)
                        chunks.append(self._build_chunk_dict(
                            file_name, page_num, chunk_counter, chunk_text, file_path, file_type
                        ))
                        chunk_counter += 1
                        current_chunk_parts = []
                        current_length = 0

                    # Slide window across long paragraph
                    start = 0
                    while start < len(para):
                        end = start + self.chunk_size_chars
                        sub_para = para[start:end]
                        chunks.append(self._build_chunk_dict(
                            file_name, page_num, chunk_counter, sub_para, file_path, file_type
                        ))
                        chunk_counter += 1
                        start += self.chunk_size_chars - self.overlap_chars
                    continue

                if current_length + len(para) > self.chunk_size_chars:
                    # Finalize current chunk
                    chunk_text = "\n\n".join(current_chunk_parts)
                    chunks.append(self._build_chunk_dict(
                        file_name, page_num, chunk_counter, chunk_text, file_path, file_type
                    ))
                    chunk_counter += 1

                    # Retain last paragraph for overlap context
                    if current_chunk_parts and len(current_chunk_parts[-1]) <= self.overlap_chars:
                        current_chunk_parts = [current_chunk_parts[-1], para]
                        current_length = len(current_chunk_parts[0]) + len(para)
                    else:
                        current_chunk_parts = [para]
                        current_length = len(para)
                else:
                    current_chunk_parts.append(para)
                    current_length += len(para)

            # Flush final chunk for the page
            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append(self._build_chunk_dict(
                    file_name, page_num, chunk_counter, chunk_text, file_path, file_type
                ))

        return chunks

    def _build_chunk_dict(
        self,
        file_name: str,
        page_number: int,
        chunk_idx: int,
        text_content: str,
        file_path: str,
        file_type: str
    ) -> Dict[str, Any]:
        chunk_id = f"{file_name}#p{page_number}#c{chunk_idx}"
        formatted_text = f"[Document: {file_name} | Page: {page_number}]\n{text_content}"
        
        return {
            "chunk_id": chunk_id,
            "doc_id": file_name,
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_type,
            "page_number": page_number,
            "chunk_index": chunk_idx,
            "text": formatted_text,
            "raw_content": text_content
        }
