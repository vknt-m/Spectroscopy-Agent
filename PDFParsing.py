"""
PDFParsing - production-ready pipeline

- Reads PDFs from:
    docs_pdfs/thesis/     (thesis logic)
    docs_pdfs/published/  (paper logic)
- Copies/renames into:
    docs_pdfs/papers/
- Chunking & parsing using pymupdf4llm + langchain_text_splitters
- Outputs Parquet dataset: pdf_chunks.parquet/
"""

import os
import re
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
import pandas as pd
import pikepdf
import pymupdf4llm
import pyarrow  # for Parquet support
import tqdm
import spacy
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from SchemaHandler import SchemaHandler






# ---------------- CONFIG ----------------
PDF_DIR = Path("docs_pdfs")
THESIS_DIR = PDF_DIR / "thesis"
PUBLISHED_DIR = PDF_DIR / "published"
OUTPUT_DIR = PDF_DIR / "papers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output will be a directory of parquet files for efficiency and scalability
OUTPUT_PARQUET_DIR = Path("pdf_chunks_parquet")
CHUNK_SIZE = 750
CHUNK_OVERLAP = 225
# ----------------------------------------

# --- SETUP NER (spaCy) ---
# The spaCy model is loaded lazily in the worker processes to avoid
# loading it multiple times in the main process.
nlp = None
# -------------------------


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
    """Ensure total filename length doesn\'t exceed OS limits (255 is max, we keep buffer)."""
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
            title = ""
            author = ""
            year = ""
            for k in ("dc:title", "pdf:Title", "Title"):
                v = meta.get(k)
                if v:
                    title = str(v).strip()
                    break
            for k in ("dc:creator", "pdf:Author", "Author"):
                v = meta.get(k)
                if v:
                    author = str(v).strip()
                    break
            if "xmp:CreateDate" in meta and meta["xmp:CreateDate"]:
                y = extract_year_from_text(str(meta["xmp:CreateDate"]))
                if y:
                    year = y
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
        try:
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

# ---------------- NER & AUTHOR FORMATTING ----------------
def extract_author_with_ner(pdf_path: Path) -> str:
    """
    Extracts author names from the first page using a NER model.
    Loads the model on first use in a worker process.
    """
    global nlp
    if nlp is None:
        try:
            # Using a smaller, more efficient model suitable for this task
            nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
            nlp = False  # Mark as failed to avoid retrying
            return ""

    if nlp is False:  # If loading failed previously
        return ""

    try:
        lines = extract_raw_lines(pdf_path, max_pages=1)
        text = "\n".join(lines)
        doc = nlp(text)

        authors = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if len(ent.text.strip().split()) > 1 and len(ent.text.strip()) < 30:
                    authors.append(ent.text.strip())

        unique_authors = list(dict.fromkeys(authors))
        if not unique_authors:
            return ""
        return ", ".join(unique_authors)
    except Exception:
        return ""

def format_author_list(author_raw: str, max_authors: int = 3) -> str:
    """
    Improved version that handles 'Last, First' format and removes affiliations.
    """
    if not author_raw:
        return "unknown"

    author_clean = re.sub(r'\(.*?\)|\[.*?\]', '', author_raw).strip()
    
    authors = []
    author_groups = re.split(r'\s+and\s+', author_clean, flags=re.IGNORECASE)
    for group in author_groups:
        authors.extend([p.strip() for p in group.split(',') if p.strip()])

    processed_authors = []
    for author in authors:
        parts = [p.strip() for p in author.split(',') if p.strip()]
        if len(parts) == 2:
            processed_authors.append(f"{parts[1]} {parts[0]}")
        elif len(parts) == 1 and len(parts[0].split()) <= 5:
             processed_authors.append(parts[0])

    if not processed_authors:
        return "unknown"

    if len(processed_authors) > max_authors:
        return ", ".join(processed_authors[:max_authors]) + " et al."
    return ", ".join(processed_authors)

def choose_concise_title(*candidates: str, min_len: int = 20, max_len: int = 50) -> str:
    cand_list = [c for c in candidates if c and c.strip()]
    if not cand_list:
        return "unknown"
    in_range = [c.strip() for c in cand_list if min_len <= len(c.strip()) <= max_len]
    if in_range:
        return min(in_range, key=lambda x: len(x))
    return min(cand_list, key=lambda x: len(x.strip()))

# ---------------- PROCESS & RENAME ----------------

def process_single_pdf(pdf_path: Path, is_thesis: bool) -> Dict[str, Any]:

    try:
        title_meta, author_meta, year_meta = merge_metadata(pdf_path)
        
        title_text, author_text = "", ""
        if is_thesis:
            lines = extract_raw_lines(pdf_path, max_pages=2)
            title_text, author_text = guess_title_and_author_from_lines(lines)
        else:
            if not title_meta:
                lines = extract_raw_lines(pdf_path, max_pages=1)
                title_text, _ = guess_title_and_author_from_lines(lines)

        final_title = choose_concise_title(title_meta, title_text, max_len=120)
        if not final_title or final_title.lower() in ("", "unknown"):
            final_title = title_meta or title_text or "unknown"

        # --- New Author Logic ---
        # 1. Try primary methods first
        author_primary_raw = author_meta or author_text
        final_author = format_author_list(author_primary_raw, max_authors=3)

        # 2. If primary methods fail, fall back to NER
        if final_author.lower() == "unknown":
            author_ner = extract_author_with_ner(pdf_path)
            if author_ner:
                final_author = format_author_list(author_ner, max_authors=3)
        
        if not final_author:
            final_author = "unknown"
        
        safe_title = sanitize_filename_part(final_title, max_len=80)
        safe_author = sanitize_filename_part(final_author, max_len=60)
        safe_year = sanitize_filename_part(year_meta or "unknown", max_len=8)

        base_filename = f"{safe_year}_{safe_title}_{safe_author}.pdf"
        base_filename = truncate_filename(base_filename)
        base_path = OUTPUT_DIR / base_filename

        if base_path.exists():
            print(f"Skipping copy: {pdf_path.name} already processed as {base_path.name}")
            selected_path = base_path
        else:
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
        for f in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
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

    snippet_hash = hashlib.md5(snippet.encode("utf-8")).hexdigest()
    for page_data in page_md_chunks:
        page_text = page_data.get("text", "")
        if hashlib.md5(page_text[:200].encode("utf-8")).hexdigest() == snippet_hash:
            return page_data.get("metadata", {}).get("page", 0) + 1

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
    
def chunk_and_save(processed_files: List[Dict[str, Any]], schema_handler: SchemaHandler):
    """
    For each processed PDF, chunk its text and save chunks to a Parquet file
    in OUTPUT_PARQUET_DIR. This uses a directory of Parquet files for scalability
    and efficient, idempotent processing.
    """
    OUTPUT_PARQUET_DIR.mkdir(exist_ok=True)

    processed_stems = {p.stem for p in OUTPUT_PARQUET_DIR.glob("*.parquet")}

    tasks = [info for info in processed_files if Path(info["new_filename"]).stem not in processed_stems]
    if not tasks:
        print("All PDFs already chunked, ensuring schema is up-to-date.")
        for info in processed_files:
            filename = info['new_filename']
            parquet_path = OUTPUT_PARQUET_DIR / f'{Path(filename).stem}.parquet'
            if parquet_path.exists():
                try:
                    df = pd.read_parquet(parquet_path)
                    info['num_chunks'] = len(df)
                    schema_handler.update_entry(info, status="processed")
                except Exception as e:
                    print(f"Warning: Could not read parquet for {filename} to update chunk count: {e}")
                    info['num_chunks'] = 0 # Set to 0 if parquet is unreadable
                    schema_handler.update_entry(info, status="processing_failed")
            else:
                info['num_chunks'] = 0 # Parquet file missing
                schema_handler.update_entry(info, status="processing_failed")
        return

    num_new_chunks = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(chunk_single_pdf, info): info for info in tasks}
        for f in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Chunking PDFs"):
            info = futures[f]
            result = f.result()
            if result:
                df = pd.DataFrame(result)
                output_path = OUTPUT_PARQUET_DIR / f"{Path(info['new_filename']).stem}.parquet"
                df.to_parquet(output_path, index=False, compression='gzip')
                # Update info with the number of chunks and set status
                info['num_chunks'] = len(df)
                schema_handler.update_entry(info, status="processed")
                num_new_chunks += len(df)
            else:
                # If chunking failed, still update the schema with 0 chunks and an error status
                info['num_chunks'] = 0
                schema_handler.update_entry(info, status="processing_failed")

    if num_new_chunks > 0:
        print(f"Saved {num_new_chunks} new chunks into {OUTPUT_PARQUET_DIR}")
    else:
        print("No new chunks created (no new PDFs to process or chunking failed).")




# ---------------- MAIN ----------------
def main():
    """
    Main pipeline:
    1. Process and copy PDFs to a clean directory.
    2. Initialize the SchemaHandler.
    3. Chunk the text of new/modified PDFs and update the schema.
    4. Save the final schema.
    """
    # Force UTF-8 for stdout and stderr to prevent encoding errors on Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    # 1. Process and copy PDFs
    processed_files = process_and_copy_pdfs()
    if not processed_files:
        print("No PDFs found to process. Exiting.")
        return

    # 2. Initialize the Schema Handler
    schema_handler = SchemaHandler()

    # 3. Chunk new files and update the schema
    chunk_and_save(processed_files, schema_handler)

    # 4. Save the final, updated schema to disk
    schema_handler.save()
    
    print("\nPDF processing and chunking complete. Schema has been updated.")

if __name__ == "__main__":
    main()
