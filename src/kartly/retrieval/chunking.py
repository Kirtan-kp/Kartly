"""Turn the knowledge-base markdown files into chunks with stable, citable ids.

One chunk is one ``##`` section of a file, so a chunk id is ``<file>#<heading-slug>`` (for example
``returns#return-window``). The policy files are written in small sections on purpose: a chunk is
the unit the retriever returns and the answer cites, so it should hold one idea. A section longer
than ``MAX_WORDS`` is split on paragraph boundaries into ``<id>`` and ``<id>-2``, ``<id>-3``...
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 200


class KnowledgeBaseError(ValueError):
    """A knowledge-base file is missing, empty or malformed."""


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str  # the file name without .md
    title: str  # the document's # heading
    heading: str  # the ## heading
    text: str

    @property
    def index_text(self) -> str:
        """What a retriever should index: the title and heading help short queries match."""
        return f"{self.title}. {self.heading}. {self.text}"


def slugify(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.casefold()).strip("-")
    return slug or "section"


def _split_long(text: str) -> list[str]:
    if len(text.split()) <= MAX_WORDS:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in re.split(r"\n\s*\n", text):
        words = len(paragraph.split())
        if current and count + words > MAX_WORDS:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        parts.append("\n\n".join(current))
    return parts


def chunk_markdown(doc: str, source: str) -> list[Chunk]:
    title = ""
    heading: str | None = None
    body: list[str] = []
    sections: list[tuple[str, str]] = []

    def close() -> None:
        if heading is not None:
            text = "\n".join(body).strip()
            if text:
                sections.append((heading, text))

    for line in source.splitlines():
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif line.startswith("## "):
            close()
            heading, body = line[3:].strip(), []
        elif heading is None:
            continue  # the synthetic-data note and anything else before the first section
        else:
            body.append(line)
    close()

    if not title:
        raise KnowledgeBaseError(f"{doc}.md has no '# ' title")
    if not sections:
        raise KnowledgeBaseError(f"{doc}.md has no '## ' sections with text")
    chunks: list[Chunk] = []
    for section_heading, text in sections:
        base = f"{doc}#{slugify(section_heading)}"
        for n, part in enumerate(_split_long(text), start=1):
            chunks.append(
                Chunk(base if n == 1 else f"{base}-{n}", doc, title, section_heading, part)
            )
    return chunks


def load_kb(directory: Path | str = "kb") -> list[Chunk]:
    """All chunks of every ``*.md`` file in ``directory``, in file then section order."""
    root = Path(directory)
    try:
        files = sorted(root.glob("*.md"))
    except OSError as exc:
        raise KnowledgeBaseError(f"cannot read {root}: {exc}") from exc
    if not files:
        raise KnowledgeBaseError(f"no *.md files in {root}")
    chunks: list[Chunk] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KnowledgeBaseError(f"cannot read {path}: {exc}") from exc
        chunks.extend(chunk_markdown(path.stem, source))
    ids = [c.id for c in chunks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise KnowledgeBaseError(f"duplicate chunk ids: {', '.join(duplicates)}")
    return chunks


def kb_digest(chunks: list[Chunk]) -> str:
    """Short fingerprint of the knowledge base, so experiment rows say which text they used."""
    material = "\n".join(f"{c.id}\n{c.text}" for c in chunks)
    return hashlib.sha256(material.encode()).hexdigest()[:8]
