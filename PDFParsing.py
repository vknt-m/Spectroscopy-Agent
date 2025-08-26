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

from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- CONFIG ----------------
PDF_DIR = Path("docs_pdfs")
THESIS_DIR = PDF_DIR / "thesis"
PUBLISHED_DIR = PDF_DIR / "published"
OUTPUT_DIR = PDF_DIR / "papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = "pdf_chunks.csv"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 225
# ----------------------------------------

# ---------------- UTILITIES ----------------
_invalid_filename_re = re.compile(r'[\\/*?:"<>|]+')
_whitespace_re = re.compile(r'\s+')
_year_re = re.compile(r'\b(19|20)\d{2}\b')

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

def truncate_filename(fullname: str, max_len: int = 180) -> str:
    """Ensure total filename length doesn't exceed OS limits (255 is max, we keep buffer)."""
    if len(fullname) <= max_len:
        return fullname
    base, ext = os.path.splitext(fullname)
    return base[: max_len - len(ext)] + ext

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
        "submitted for", "a thesis", "by", "author", "supervisor", "right",
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

def process_single_pdf(pdf_path: Path, is_thesis: bool) -> Dict[str, Any]:

    try:
        title_meta, author_meta, year_meta = merge_metadata(pdf_path)
        # If thesis, apply thesis logic to get title/author from page content

        title_text, author_text = "", ""
        if is_thesis:
            lines = extract_raw_lines(pdf_path, max_pages=2)
            title_text, author_text = guess_title_and_author_from_lines(lines)
        else:
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
        safe_title = sanitize_filename_part(final_title, max_len=80)
        safe_author = sanitize_filename_part(final_author, max_len=60)
        safe_year = sanitize_filename_part(year_meta or "unknown", max_len=8)

        # Build base filename and prefer skipping copy if already present (idempotency)
        base_filename = f"{safe_year}_{safe_title}_{safe_author}.pdf"
        base_filename = truncate_filename(base_filename)
        base_path = OUTPUT_DIR / base_filename

        if base_path.exists():
            # Already processed previously — reuse existing file (skip re-copy)
            print(f"Skipping copy: {pdf_path.name} already processed as {base_path.name}")
            selected_path = base_path
        else:
            # Ensure unique filename in case of race with other processes, then copy
            selected_path = ensure_unique_filename(base_path)
            try:
                shutil.copy2(pdf_path, selected_path)
            except Exception as e:
                return {"orig_path": str(pdf_path), "error": str(e)}

        return {
            "orig_path": str(pdf_path),
            "new_filename": selected_path.name,
            "title": final_title,
            "author": final_author,
            "year": safe_year,
            "type": "thesis" if is_thesis else "published"
        }
    except Exception as e:
        return {"orig_path": str(pdf_path), "error": str(e)}

def process_and_copy_pdfs() -> List[Dict[str, Any]]:
    """
    Process thesis and published folders (by folder) and /rename into OUTPUT_DIR.
    Returns list of dicts: { orig_path, new_filename, title, author, year, type }
    """
    tasks = []
    for folder, is_thesis in [(THESIS_DIR, True), (PUBLISHED_DIR, False)]:
        if folder.exists():
            for pdf_path in folder.glob("*.pdf"):
                tasks.append((pdf_path, is_thesis))

    processed = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_pdf, p, is_thesis): (p, is_thesis) for p, is_thesis in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            result = f.result()
            if result and "error" not in result:
                processed.append(result)
            else:
                print(f"Warning: failed processing {result.get('orig_path')}: {result.get('error')}")

    return processed

# ---------------- CHUNKING ----------------
def get_closest_page_number(chunk_text: str, page_md_chunks: List[Dict[str, Any]]) -> int:
    """Finds 1-based page where chunk_text starts; fallback to 1.
    Optimized: uses hashing + fallback fuzzy search for speed."""
    import hashlib
    from difflib import SequenceMatcher

    if not chunk_text or not page_md_chunks:
        return 0
    snippet = chunk_text[:200].strip()
    if not snippet:
        return 0

    # Precompute hashes of page texts
    snippet_hash = hashlib.md5(snippet.encode("utf-8")).hexdigest()
    for page_data in page_md_chunks:
        page_text = page_data.get("text", "")
        if hashlib.md5(page_text[:200].encode("utf-8")).hexdigest() == snippet_hash:
            return page_data.get("metadata", {}).get("page", 0) + 1

    # Fallback: fuzzy matching on first 30 chars
    best_score, best_page = 0.0, 1
    for page_data in page_md_chunks:
        page_text = page_data.get("text", "")
        ratio = SequenceMatcher(None, snippet[:30], page_text[:200]).ratio()
        if ratio > best_score:
            best_score = ratio
            best_page = page_data.get("metadata", {}).get("page", 0) + 1
    return best_page


def chunk_single_pdf(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    pdf_filename = info["new_filename"]
    pdf_path = OUTPUT_DIR / pdf_filename
    if not pdf_path.exists():
        return []
    try:
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        page_md_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        full_md = "\n\n".join([p.get("text", "") for p in page_md_chunks])
        header_docs = markdown_splitter.split_text(full_md)
        final_docs = text_splitter.split_documents(header_docs)
        chunks = []
        for doc in final_docs:
            page_num = get_closest_page_number(doc.page_content, page_md_chunks)
            chunks.append({
                "source_filename": pdf_filename,
                "title": info["title"],
                "author": info["author"],
                "year": info["year"],
                "page_number": page_num,
                "chunk_text": doc.page_content,
                "chunk_metadata": str(getattr(doc, "metadata", {}))
            })
        return chunks
    except Exception as e:
        import traceback
        with open("errors.log", "a", encoding="utf-8") as logf:
            logf.write(f"Failed chunking {pdf_filename}: {e}\n")
            logf.write(traceback.format_exc() + "\n")
        print(f"Warning: failed chunking {pdf_filename}, see errors.log for details.")
        return []
    
def chunk_and_save(processed_files: List[Dict[str, Any]]):
    """
    For each processed PDF, chunk its text and save all chunks to OUTPUT_CSV_PATH.
    Each chunk has metadata: source_filename, title, author, year, page_number, chunk_text, chunk_metadata
    """
    header_written = Path(OUTPUT_CSV_PATH).exists()

    # Load already-processed filenames from existing CSV, if present
    processed_filenames = set()
    if header_written:
        try:
            existing_df = pd.read_csv(OUTPUT_CSV_PATH, usecols=["source_filename"])
            processed_filenames = set(existing_df["source_filename"].unique())
        except Exception:
            # if reading fails for any reason, continue with empty set (will attempt chunking)
            processed_filenames = set()

    # Only chunk files that are not already present in the CSV
    tasks = [info for info in processed_files if info["new_filename"] not in processed_filenames]
    if not tasks:
        print("All PDFs already chunked, skipping.")
        return

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(chunk_single_pdf, info): info for info in tasks}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Chunking PDFs"):
            result = f.result()
            if result:
                # incremental CSV writing instead of keeping all in memory
                df = pd.DataFrame(result)
                df.to_csv(
                    OUTPUT_CSV_PATH,
                    mode="a",
                    header=not header_written,
                    index=False,
                    encoding="utf-8"
                )
                header_written = True

    if header_written:
        print(f"Chunks appended to {OUTPUT_CSV_PATH}")
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
