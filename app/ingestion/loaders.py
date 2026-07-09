"""Document loaders for supported support-knowledge source formats."""

from pathlib import Path
from typing import Final

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader

SUPPORTED_EXTENSIONS: Final[set[str]] = {".pdf", ".md", ".markdown", ".txt", ".html", ".htm"}


def _html_to_plain_text_with_headings(html: str) -> str:
    """Convert HTML to plain text while preserving headings as markdown markers.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with heading elements rendered as ``#``-prefixed lines.
    """
    soup = BeautifulSoup(html, "html.parser")

    for unwanted in soup(["script", "style", "noscript"]):
        unwanted.decompose()

    lines: list[str] = []

    def walk(node: Tag | NavigableString) -> None:
        """Walk a BeautifulSoup node and append meaningful text lines."""
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                lines.append(text)
            return

        if not isinstance(node, Tag):
            return

        if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(node.name[1])
            heading_text = node.get_text(" ", strip=True)
            if heading_text:
                lines.append(f"{'#' * level} {heading_text}")
            return

        if node.name in {"p", "li", "blockquote"}:
            text = node.get_text(" ", strip=True)
            if text:
                lines.append(text)
            return

        for child in node.children:
            walk(child)

    body = soup.body or soup
    walk(body)

    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines).strip()


def load_pdf(path: Path) -> str:
    """Extract text from a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted plain text.
    """
    reader = PdfReader(path)
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(text.strip() for text in page_text if text.strip()).strip()


def load_markdown(path: Path) -> str:
    """Load markdown and convert it to plain text with heading markers preserved.

    Args:
        path: Path to the markdown file.

    Returns:
        Plain text representation of the markdown document.
    """
    raw_markdown = path.read_text(encoding="utf-8")
    html = markdown.markdown(raw_markdown, extensions=["extra", "sane_lists"])
    return _html_to_plain_text_with_headings(html)


def load_txt(path: Path) -> str:
    """Load a UTF-8 plain-text document.

    Args:
        path: Path to the text file.

    Returns:
        Plain text document content.
    """
    return path.read_text(encoding="utf-8").strip()


def load_html(path: Path) -> str:
    """Load HTML and strip it to plain text with heading markers preserved.

    Args:
        path: Path to the HTML file.

    Returns:
        Plain text representation of the HTML document.
    """
    html = path.read_text(encoding="utf-8")
    return _html_to_plain_text_with_headings(html)


def load_document(path: Path) -> str:
    """Load a supported document by dispatching on file extension.

    Args:
        path: Path to the source document.

    Raises:
        ValueError: If the document extension is unsupported.

    Returns:
        Extracted plain text content.
    """
    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)
    if extension in {".md", ".markdown"}:
        return load_markdown(path)
    if extension == ".txt":
        return load_txt(path)
    if extension in {".html", ".htm"}:
        return load_html(path)

    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(f"Unsupported document type '{extension}'. Supported types: {supported}.")
