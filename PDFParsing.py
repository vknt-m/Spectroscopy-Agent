#!/usr/bin/env python3
"""
PDFParsing - production-ready pipeline

- Reads PDFs from:
    docs_pdfs/thesis/     (thesis logic)
    docs_pdfs/published/  (paper logic)
- Copies/renames into:
    docs_pdfs/papers/
- Chunking & parsing using pymupdf4llm + langchain_text_splitters
- Outputs CSV: pdf_chunks.csv
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import pikepdf
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import pandas as pd
from tqdm import tqdm

# ---------------- CONFIG ----------------
PDF_DIR = Path("docs_pdfs")
THESIS_DIR = PDF_DIR / "thesis"
PUBLISHED_DIR = PDF_DIR / "published"
OUTPUT_DIR = PDF_DIR / "papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = "pdf_chunks.csv"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# ----------------------------------------

# ---------------- UTILITIES ----------------
_invalid_filename_re = re.compile(r'[\\/*?:"<>|]+')
_whitespace_re = re.compile(r'\s+')
_year_re = re.compile(r'(19|20)\d{2}')

def sanitize_filename_part(s: str, max_len: int = 120) -> str:
    if not s:
        return "unknown"
    s = str(s)
    s = _invalid_filename_re.sub("", s)
    s = _whitespace_re.sub(" ", s).strip()
    s = s.replace(",", "")  # remove commas for file clarity
    if not s:
        return "unknown"
    return s[:max_len]

def ensure_unique_filename(path: Path) -> Path:
    """Append counter if file exists."""
    if not path.exists():
        return path
    base = path.stem
    suf = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{base}_{counter}{suf}")
        if not candidate.exists():
            return candidate
        counter += 1

def extract_year_from_text(text: str) -> str:
    if not text:
        return ""
    m = _year_re.search(text)
    if m:
        return m.group(0)
    return ""

# ---------------- METADATA EXTRACTION (both tools) ----------------
def extract_metadata_pymupdf(pdf_path: Path) -> Dict[str, str]:
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        creation = meta.get("creationDate") or meta.get("modDate") or ""
        # PyMuPDF creationDate often like "D:YYYYMMDD..."
        year = ""
        if isinstance(creation, str):
            m = re.search(r"D:(\d{4})", creation)
            if m:
                year = m.group(1)
            else:
                year = extract_year_from_text(creation)
        doc.close()
        return {"title": title, "author": author, "year": year}
    except Exception:
        return {"title": "", "author": "", "year": ""}

def extract_metadata_pikepdf(pdf_path: Path) -> Dict[str, str]:
    try:
        with pikepdf.open(pdf_path) as pdf:
            meta = pdf.open_metadata()
            # XMP keys vary by file — try common ones
            # dc:title or pdf:Title or title
            title = ""
            author = ""
            year = ""
            # pikepdf metadata mapping: meta.get("dc:title") etc.
            for k in ("dc:title", "pdf:Title", "Title", "dc:Title"):
                v = meta.get(k)
                if v:
                    title = str(v).strip()
                    break
            for k in ("dc:creator", "pdf:Author", "Author", "dc:creator"):
                v = meta.get(k)
                if v:
                    author = str(v).strip()
                    break
            if "xmp:CreateDate" in meta and meta["xmp:CreateDate"]:
                y = extract_year_from_text(str(meta["xmp:CreateDate"]))
                if y:
                    year = y
            # fallback look in raw docinfo too
            if not title or not author or not year:
                try:
                    docinfo = pdf.docinfo
                    if not title:
                        t = docinfo.get("/Title")
                        if t:
                            title = str(t)
                    if not author:
                        a = docinfo.get("/Author")
                        if a:
                            author = str(a)
                    if not year:
                        c = docinfo.get("/CreationDate") or docinfo.get("/ModDate")
                        if c:
                            year = extract_year_from_text(str(c))
                except Exception:
                    pass
            return {"title": (title or "").strip(), "author": (author or "").strip(), "year": (year or "").strip()}
    except Exception:
        return {"title": "", "author": "", "year": ""}

def merge_metadata(pdf_path: Path) -> Tuple[str, str, str]:
    """
    Merge metadata from pikepdf (prefer) and pymupdf as fallback.
    Also try to infer year from filename if none found.
    Returns title, author, year (strings, maybe empty).
    """
    meta_pike = extract_metadata_pikepdf(pdf_path)
    meta_mu = extract_metadata_pymupdf(pdf_path)

    title = meta_pike.get("title") or meta_mu.get("title") or ""
    author = meta_pike.get("author") or meta_mu.get("author") or ""
    year = meta_pike.get("year") or meta_mu.get("year") or ""

    if not year:
        year = extract_year_from_text(pdf_path.name) or "unknown"

    return title, author, year

# ---------------- THESIS LOGIC (unchanged core) ----------------
def extract_raw_lines(pdf_path: Path, max_pages: int = 2) -> List[str]:
    doc = fitz.open(pdf_path)
    lines: List[str] = []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        text = page.get_text("text")
        page_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines.extend(page_lines)
    doc.close()
    return lines

def guess_title_and_author_from_lines(lines: List[str]) -> Tuple[str, str]:
    title = ""
    author = ""

    title_stop_phrases = [
        "submitted for", "a thesis", "by", "author", "supervisor", "copyright",
        "indian institute", "faculty of", "department of", "centre for", "chapter", "section"
    ]

    max_title_lines = 5
    title_buffer = []
    idx = 0
    while idx < min(len(lines), 40) and len(title_buffer) < max_title_lines:
        line = lines[idx]
        if len(line) < 8 or len(line.split()) < 2:
            idx += 1
            continue
        if any(phrase in line.lower() for phrase in title_stop_phrases):
            break
        title_buffer.append(line)
        idx += 1

    if title_buffer:
        title = " ".join(title_buffer)

    # Author detection
    for i, line in enumerate(lines[:60]):
        if re.match(r'^\s*(by|author)\s*:?\s*$', line, re.I):
            for off in range(1, 4):
                if i + off < len(lines):
                    cand = lines[i + off]
                    if 2 <= len(cand.split()) <= 5 and not any(c in cand for c in "@:/\\"):
                        author = cand
                        break
            if author:
                break
        elif re.match(r'^\s*(by|author)\s*:?\s*\w+', line, re.I):
            candidate = re.sub(r'^\s*(by|author)\s*:?\s*', '', line, flags=re.I).strip()
            if 2 <= len(candidate.split()) <= 5:
                author = candidate
                break

    if not author and title:
        # fallback: look 10 lines after first title line (title_buffer[0])
        try:
            # find index of first title_buffer line in lines
            title_first = title_buffer[0]
            title_idx = lines.index(title_first)
            for j in range(title_idx + 1, min(title_idx + 11, len(lines))):
                cand = lines[j]
                if 2 <= len(cand.split()) <= 5 and cand[0].isupper() and not any(c in cand for c in ":@/\\"):
                    author = cand
                    break
        except Exception:
            pass

    return title.strip(), author.strip()

# ---------------- AUTHOR / TITLE FORMATTING HELPERS ----------------
def compact_author_list(author_raw: str, max_authors: int = 3) -> str:
    if not author_raw:
        return "unknown"
    # split on commas or " and "
    parts = [p.strip() for p in re.split(r',| and ', author_raw) if p.strip()]
    # filter out obviously non-name parts
    cleaned = []
    for p in parts:
        # remove affiliation-like phrases (heuristic)
        if len(p.split()) > 1 and re.search(r'[A-Za-z]', p):
            cleaned.append(p)
    if not cleaned:
        return "unknown"
    if len(cleaned) > max_authors:
        return ", ".join(cleaned[:max_authors]) + " et al."
    return ", ".join(cleaned)

def choose_concise_title(*candidates: str, min_len: int = 20, max_len: int = 50) -> str:
    cand_list = [c for c in candidates if c and c.strip()]
    if not cand_list:
        return "unknown"
    # prefer those within desired length
    in_range = [c.strip() for c in cand_list if min_len <= len(c.strip()) <= max_len]
    if in_range:
        # pick shortest in range (more concise)
        return min(in_range, key=lambda x: len(x))
    # fallback to shortest non-empty
    return min(cand_list, key=lambda x: len(x.strip()))

# ---------------- PROCESS & RENAME ----------------
def process_and_copy_pdfs() -> List[Dict[str, Any]]:
    """
    Process thesis and published folders (by folder) and copy/rename into OUTPUT_DIR.
    Returns list of dicts: { orig_path, new_filename, title, author, year, type }
    """
    processed = []

    folders = [
        (THESIS_DIR, True),
        (PUBLISHED_DIR, False)
    ]

    for folder, is_thesis in folders:
        if not folder.exists():
            continue
        for pdf_path in folder.glob("*.pdf"):
            # read metadata from both tools
            title_meta_mu, author_meta_mu, year_meta_mu = "", "", ""
            try:
                m1 = extract_metadata_pymupdf(pdf_path)
                title_meta_mu, author_meta_mu, year_meta_mu = m1.get("title", ""), m1.get("author", ""), m1.get("year", "")
            except Exception:
                title_meta_mu, author_meta_mu, year_meta_mu = "", "", ""

            title_meta_pike, author_meta_pike, year_meta_pike = "", "", ""
            try:
                mp = extract_metadata_pikepdf(pdf_path)
                title_meta_pike, author_meta_pike, year_meta_pike = mp.get("title", ""), mp.get("author", ""), mp.get("year", "")
            except Exception:
                title_meta_pike, author_meta_pike, year_meta_pike = "", "", ""

            # merged metadata
            title_meta = title_meta_pike or title_meta_mu or ""
            author_meta = author_meta_pike or author_meta_mu or ""
            year_meta = year_meta_pike or year_meta_mu or ""
            if not year_meta:
                year_meta = extract_year_from_text(pdf_path.name) or "unknown"

            # If thesis, apply thesis logic to get title/author from page content
            title_text = ""
            author_text = ""
            if is_thesis:
                lines = extract_raw_lines(pdf_path, max_pages=2)
                title_text, author_text = guess_title_and_author_from_lines(lines)
            else:
                # published: if title missing in metadata, try title extraction
                if not title_meta:
                    lines = extract_raw_lines(pdf_path, max_pages=1)
                    title_text, _ = guess_title_and_author_from_lines(lines)

            # final selection (concise title preference)
            final_title = choose_concise_title(title_meta, title_text, max_len=120)
            if not final_title or final_title.lower() in ("", "unknown"):
                final_title = title_meta or title_text or "unknown"

            # authors: prefer metadata if it's a clean name list, otherwise use text-extracted
            final_author_raw = author_meta or author_text or "unknown"
            final_author = compact_author_list(final_author_raw, max_authors=3)

            # limit lengths for filename components
            safe_title = sanitize_filename_part(final_title, max_len=80)
            safe_author = sanitize_filename_part(final_author, max_len=60)
            safe_year = sanitize_filename_part(year_meta or "unknown", max_len=8)

            new_filename = f"{safe_year}_{safe_title}_{safe_author}.pdf"
            new_path = OUTPUT_DIR / new_filename
            new_path = ensure_unique_filename(new_path)

            # copy file (preserve original)
            shutil.copy2(pdf_path, new_path)

            processed.append({
                "orig_path": str(pdf_path),
                "new_filename": new_path.name,
                "title": final_title,
                "author": final_author,
                "year": safe_year,
                "type": "thesis" if is_thesis else "published"
            })
    return processed

# ---------------- CHUNKING ----------------
def get_closest_page_number(chunk_text: str, page_md_chunks: List[Dict[str, Any]]) -> int:
    """Finds 1-based page where chunk_text starts; fallback to 1."""
    if not chunk_text or not page_md_chunks:
        return 0
    snippet = chunk_text[:200].strip()
    if not snippet:
        return 0
    for page_data in page_md_chunks:
        page_text = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0) + 1
        if snippet in page_text:
            return page_num
    # fallback: search by first few words
    first_words = " ".join(snippet.split()[:5])
    for page_data in page_md_chunks:
        page_text = page_data.get("text", "")
        page_num = page_data.get("metadata", {}).get("page", 0) + 1
        if first_words and first_words in page_text:
            return page_num
    return 1

def chunk_and_save(processed_files: List[Dict[str, Any]]):
    all_chunks = []

    # init splitters
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for info in tqdm(processed_files, desc="Chunking PDFs"):
        pdf_filename = info["new_filename"]
        pdf_path = OUTPUT_DIR / pdf_filename
        if not pdf_path.exists():
            continue

        try:
            # convert to markdown with page chunks
            page_md_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            full_md = "\n\n".join([p.get("text", "") for p in page_md_chunks])

            header_docs = markdown_splitter.split_text(full_md)
            final_docs = text_splitter.split_documents(header_docs)

            for doc in final_docs:
                page_num = get_closest_page_number(doc.page_content, page_md_chunks)
                chunk_record = {
                    "source_filename": pdf_filename,
                    "title": info["title"],
                    "author": info["author"],
                    "year": info["year"],
                    "page_number": page_num,
                    "chunk_text": doc.page_content,
                    "chunk_metadata": str(getattr(doc, "metadata", {}))
                }
                all_chunks.append(chunk_record)

        except Exception as e:
            print(f"Warning: failed chunking {pdf_filename}: {e}")

    # save CSV if we have chunks
    if all_chunks:
        df = pd.DataFrame(all_chunks)
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
        print(f"Saved {len(df)} chunks to {OUTPUT_CSV_PATH}")
    else:
        print("No chunks created (no PDFs processed or chunking failed).")

# ---------------- MAIN ----------------
def main():
    processed_files = process_and_copy_pdfs()
    if not processed_files:
        print("No PDFs processed. Check folders.")
        return
    chunk_and_save(processed_files)
    print("Done.")

if __name__ == "__main__":
    main()
