import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    page_num: int
    chunk_id: str
    start_char: int
    end_char: int

@dataclass
class Page:
    text: str
    page_num: int
    page_id: str
    word_count: int

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        print(f"[PDF] Loaded '{pdf_path}' — {self.total_pages} pages")

    def extract_pages(self) -> List[Page]:
        """Extract text page-by-page (used by Page Index RAG)."""
        pages = []
        for page_num in range(self.total_pages):
            page = self.doc[page_num]
            text = page.get_text("text").strip()
            if not text:   # skip blank pages
                continue
            pages.append(Page(
                text=text,
                page_num=page_num + 1,
                page_id=f"page_{page_num + 1}",
                word_count=len(text.split())
            ))
        print(f"[PDF] Extracted {len(pages)} non-empty pages")
        return pages

    def extract_chunks(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Chunk]:
        """
        Extract sliding-window chunks (used by VectorDB RAG).
        chunk_size and chunk_overlap are measured in characters.
        """
        chunks = []
        chunk_index = 0

        for page_num in range(self.total_pages):
            page = self.doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue

            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()

                if len(chunk_text) > 50:   # skip tiny fragments
                    chunks.append(Chunk(
                        text=chunk_text,
                        page_num=page_num + 1,
                        chunk_id=f"chunk_{chunk_index}",
                        start_char=start,
                        end_char=end
                    ))
                    chunk_index += 1

                if end >= len(text):
                    break
                start = end - chunk_overlap  # sliding window overlap

        print(f"[PDF] Created {len(chunks)} chunks "
              f"(size={chunk_size}, overlap={chunk_overlap})")
        return chunks