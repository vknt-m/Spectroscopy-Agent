
import argparse
import sys
import shutil
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---
# This is the target directory where new PDFs will be placed for processing.
# It must match the directory monitored by PDFParsing.py
PUBLISHED_DIR = Path("docs_pdfs/published")
# ---------------------

def run_command(command: list[str], description: str):
    """Runs a command as a subprocess and checks for errors."""
    print(f"\n--- {description} ---")
    try:
        # Using sys.executable ensures we use the same Python interpreter
        # that is running this script.
        # The output will be streamed directly to the console in real-time.
        process = subprocess.run(
            [sys.executable] + command,
            check=True,        # Raises CalledProcessError if the command returns a non-zero exit code.
        )
        print(f"--- Successfully completed: {description} ---")
        return True
    except FileNotFoundError:
        print(f"Error: The command '{command[0]}' was not found.")
        print("Please ensure the script exists and is in the correct path.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"--- Error during: {description} ---")
        print(f"Return Code: {e.returncode}")
        # Stdout and Stderr from the failed process are already printed to the console.
        print(f"--- Failed: {description} ---")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during '{description}': {e}")
        return False

def get_pdf_files_from_dialog():
    """Opens a file dialog to select one or more PDF files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    filepaths = filedialog.askopenfilenames(
        title="Select PDF files to process",
        filetypes=[("PDF Files", "*.pdf"), ("All files", "*.*")]
    )
    return [Path(fp) for fp in filepaths]

def main():
    """
    Orchestrates the entire pipeline:
    1. Copies user-provided PDFs to the processing directory.
    2. Runs the PDF parsing and chunking script.
    3. Runs the embedding script to add the new content to the database.
    """
    parser = argparse.ArgumentParser(
        description="Run the full ingestion pipeline for one or more PDF files.",
        epilog="Example: python run_pipeline.py my_document.pdf another_paper.pdf"
    )
    parser.add_argument(
        "pdf_files",
        nargs="*",  # Makes the argument optional, accepting zero or more files
        default=[],
        type=Path,
        help="One or more paths to the PDF files you want to add. If none are provided, a file dialog will open."
    )
    args = parser.parse_args()

    pdf_files_to_process = args.pdf_files
    
    # If no files were passed via CLI, open the file dialog
    if not pdf_files_to_process:
        print("No PDF files provided via command line. Opening file selection dialog...")
        pdf_files_to_process = get_pdf_files_from_dialog()

    if not pdf_files_to_process:
        print("No files selected. Exiting.")
        return

    # --- 1. Copy PDF files ---
    print("\n--- Stage 1: Copying PDF files ---")
    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    for pdf_path in pdf_files_to_process:
        if not pdf_path.exists() or not pdf_path.is_file():
            print(f"Warning: File not found or is not a file, skipping: {pdf_path}")
            continue
        
        destination_path = PUBLISHED_DIR / pdf_path.name
        print(f"Copying '{pdf_path}' to '{destination_path}'...")
        shutil.copy2(pdf_path, destination_path)
        copied_files += 1
    
    if copied_files == 0:
        print("\nNo valid files were provided or copied. Exiting.")
        return

    print(f"--- Successfully copied {copied_files} file(s). ---")

    # --- 2. Run PDF Parsing and Chunking ---
    if not run_command(["PDFParsing.py"], "Stage 2: Parsing and Chunking PDFs"):
        print("\nHalting pipeline due to error in PDF parsing stage.")
        return

    # --- 3. Run Embedding ---
    if not run_command(["Embedding.py"], "Stage 3: Generating and Storing Embeddings"):
        print("\nHalting pipeline due to error in embedding stage.")
        return

    print("\n\n==========================================")
    print("Pipeline completed successfully!")
    print("Your new documents have been processed and added to the vector database.")
    print("==========================================")


if __name__ == "__main__":
    main()
