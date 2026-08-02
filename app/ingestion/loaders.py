"""Multi-format document loader. Accepts .md, .txt, .html, .pdf and
normalizes each into a list of Document objects with clean plaintext +
metadata (source file, section heading, page number where applicable).

Raw files are left untouched in data/raw; normalized text goes to
data/processed so re-indexing never requires re-uploading source files.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import PROCESSED_DIR


@dataclass
class Document:
    doc_id: str  # stable id derived from filename
    source_file: str
    text: str
    section_heading: Optional[str] = None
    page_number: Optional[int] = None
    extra: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_markdown(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    docs: list[Document] = []
    # Split on top-level headings so each section carries its own heading
    # metadata; this feeds the "recursive/structure-aware" chunker later.
    sections = re.split(r"(?m)^(#{1,3} .+)$", raw)
    if sections[0].strip():
        docs.append(Document(doc_id=path.stem, source_file=path.name,
                              text=_clean_text(sections[0]), section_heading=None))
    for i in range(1, len(sections), 2):
        heading = sections[i].lstrip("#").strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        text = _clean_text(body)
        if text:
            docs.append(Document(doc_id=path.stem, source_file=path.name,
                                  text=text, section_heading=heading))
    return docs


def load_text(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(doc_id=path.stem, source_file=path.name, text=_clean_text(raw))]


def load_html(path: Path) -> list[Document]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    docs: list[Document] = []
    # Walk headings to preserve section structure, similar to markdown.
    current_heading = None
    buffer: list[str] = []

    def flush():
        text = _clean_text(" ".join(buffer))
        if text:
            docs.append(Document(doc_id=path.stem, source_file=path.name,
                                  text=text, section_heading=current_heading))
        buffer.clear()

    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            flush()
            current_heading = el.get_text(strip=True)
        else:
            buffer.append(el.get_text(" ", strip=True))
    flush()
    if not docs:  # fallback: no structured tags found
        docs.append(Document(doc_id=path.stem, source_file=path.name,
                              text=_clean_text(soup.get_text(" "))))
    return docs


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(doc_id=path.stem, source_file=path.name,
                                  text=text, page_number=page_num))
    return docs


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext} ({path.name})")
    return LOADERS[ext](path)


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in LOADERS:
            docs.extend(load_file(path))
    return docs


def persist_processed(docs: list[Document]) -> Path:
    """Write normalized docs to data/processed as JSONL so the ingestion
    pipeline can be re-run (re-chunk / re-embed) without touching raw files."""
    out_path = PROCESSED_DIR / "documents.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")
    return out_path


def load_processed() -> list[Document]:
    out_path = PROCESSED_DIR / "documents.jsonl"
    docs = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            docs.append(Document(**json.loads(line)))
    return docs
