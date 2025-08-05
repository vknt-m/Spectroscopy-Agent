
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import os
import pandas as pd
from tqdm import tqdm
import re # For regex for title cleaning
from typing import List, Dict, Any

# --- Configuration ---
PDF_DIR = "docs_pdfs" # The folder with your downloaded PDFs
OUTPUT_CSV_PATH = "spectroscopy_chunks_final_robust.csv"
CHUNK_SIZE = 1000 # The target size for each final chunk in characters
CHUNK_OVERLAP = 150 # The overlap between consecutive final chunks

# --- Helper Functions ---
def safe_filename(name):
    """Sanitizes a string to be safe for filenames."""
    return "".join(c if c.isalnum() or c in " ._-" else "_" for c in str(name))

def get_closest_page_number(chunk_text: str, page_md_chunks: List[Dict[str, Any]]) -> int:
    """
    Finds the 1-based page number where the majority or start of the chunk_text is located.
    This is the most critical part for accurate page citation.
    """
    if not chunk_text or not page_md_chunks:
        return 0 # Default to 0 or unknown page

    # Look for the start of the chunk in each page's content
    search_snippet = chunk_text[:min(200, len(chunk_text))].strip() # Use first N chars

    # Try to find an exact match of the snippet's presence
    for page_chunk_data in page_md_chunks:
        page_md = page_chunk_data['text']
        page_num = page_chunk_data['metadata'].get('page', 0) + 1 # Convert to 1-based
        if search_snippet and search_snippet in page_md:
            return page_num
    
    # Fallback: If exact snippet not found (e.g., due to overlap cutting it off),
    # find the page where the first word(s) appear.
    first_words = " ".join(search_snippet.split()[:5]) # Take first few words
    if first_words:
        for page_chunk_data in page_md_chunks:
            page_md = page_chunk_data['text']
            page_num = page_chunk_data['metadata'].get('page', 0) + 1
            if first_words in page_md:
                return page_num

    # Ultimate Fallback: If nothing matches, assign to page 1 or 0
    return 1 if page_md_chunks else 0 # Default to page 1 if document exists


# --- Main Parsing and Chunking Logic ---
all_chunks_data = []

if not os.path.isdir(PDF_DIR):
    print(f"Error: Directory '{PDF_DIR}' not found. Please ensure your PDFs are in this folder.")
else:
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to process in '{PDF_DIR}'.")

    # --- Initialize the Splitters ---
    # Structural splitter for headers. Its split_text() method returns a list of Document objects.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    
    # Size-based splitter for final chunks. Its split_documents() method accepts Document objects.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs and Chunking"):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        
        try:
            # --- Metadata Extraction from Filename ---
            base_name = pdf_file[:-4]
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                metadata_part, md5 = parts
                meta_parts = metadata_part.rsplit('_', 2)
                if len(meta_parts) == 3:
                    title, author, year = meta_parts
                else:
                    title, author, year = metadata_part, "unknown", "unknown"
            else:
                title, author, year = base_name, "unknown", "unknown"

            # --- Step 1: Convert Entire PDF to Markdown, retaining page data ---
            # page_md_chunks is a list of dictionaries, one per page: [{'text': '...', 'metadata': {'page': 0, ...}}, ...]
            page_md_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            
            # Combine all page Markdown into a single string for structural splitting
            full_doc_md_content = "\n\n".join([page_data['text'] for page_data in page_md_chunks])

            # --- Step 2: Apply Structural Splitting to the entire document Markdown ---
            # markdown_splitter.split_text() returns a list of LangChain Document objects directly.
            header_based_docs = markdown_splitter.split_text(full_doc_md_content)

            # --- Step 3: Apply Size-based Splitting and Assign Page Numbers ---
            # text_splitter.split_documents() accepts a list of Document objects.
            # This will create the final chunks, propagating metadata.
            final_chunks_docs = text_splitter.split_documents(header_based_docs)

            for final_doc in final_chunks_docs:
                # --- Page Number Lookup Logic ---
                # Find the page number for this final chunk.
                found_page_number = get_closest_page_number(final_doc.page_content, page_md_chunks)
                
                chunk_data = {
                    "source_filename": pdf_file,
                    "title": title,
                    "author": author,
                    "year": year,
                    "page_number": found_page_number, # The page number we looked up
                    "chunk_text": final_doc.page_content,
                    # Store all metadata propagated from the splitters
                    "chunk_metadata": str(final_doc.metadata) 
                }
                all_chunks_data.append(chunk_data)

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

# --- Save to CSV ---
if all_chunks_data:
    df = pd.DataFrame(all_chunks_data)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print(f"\nSuccessfully processed {len(df['source_filename'].unique())} PDFs.")
    print(f"Created {len(df)} structured Markdown chunks and saved to '{OUTPUT_CSV_PATH}'.")
else:
    print("\nNo chunks were created. Please check your PDF directory and file contents.")






