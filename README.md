# Spectroscopy Agent

The Spectroscopy Agent is a research assistant that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions about spectroscopy documents. It can process PDF files, extract information, and use that information to answer user queries through a command-line interface or a web-based chat interface.

## Features

- **PDF Processing:** Ingests PDF documents (theses and published papers) and processes them for information retrieval.
- **Metadata Extraction:** Extracts metadata such as title, author, and year from the PDFs.
- **Text Chunking:** Splits the text of the documents into smaller, manageable chunks.
- **Vector Embeddings:** Generates vector embeddings for the text chunks using a sentence transformer model.
- **Information Retrieval:** Retrieves relevant text chunks from a vector database based on user queries.
- **Re-ranking:** (Optional) Re-ranks search results for improved relevance using a CrossEncoder model.
- **Conversational AI:** Uses a Large Language Model (LLM) to generate answers based on the retrieved information.
- **User Interfaces:** Provides both a command-line interface (CLI) and a web-based interface (Gradio) for interacting with the agent.
- **State Management:** Keeps track of the processing state of each document using a schema file.

## How it Works

The Spectroscopy Agent is built around a RAG pipeline that consists of two main stages: data ingestion and retrieval/generation.

### 1. Data Ingestion

The data ingestion pipeline is orchestrated by the `run_pipeline.py` script. It takes PDF files as input and performs the following steps:

1.  **Copying:** The PDF files are copied to the `docs_pdfs/published` directory.
2.  **Parsing and Chunking (`PDFParsing.py`):
    -   The script reads PDFs from the `docs_pdfs/thesis` and `docs_pdfs/published` directories.
    -   It extracts metadata (title, author, year) from the PDFs using `PyMuPDF` and `pikepdf`.
    -   It renames the PDFs based on the extracted metadata and copies them to the `docs_pdfs/papers` directory.
    -   The text of the PDFs is then split into smaller chunks, which are saved as Parquet files in the `pdf_chunks_parquet` directory.
    -   The status of each processed file is tracked in the `info_schema.json` file.
3.  **Embedding (`Embedding.py`):
    -   This script reads the chunked Parquet files.
    -   It generates vector embeddings for the text chunks using the `BAAI/bge-small-en-v1.5` sentence transformer model.
    -   The embeddings are stored in a ChromaDB vector database located in the `.spectroscopy_chromadb` directory.
    -   The process is idempotent, meaning it avoids duplicating chunks that are already in the database.

### 2. Retrieval and Generation

The retrieval and generation stage is handled by the main application (`app.py`), which can be run as a CLI or a Gradio web app.

1.  **User Query:** The user enters a query through the CLI or the web interface.
2.  **Retrieval (`Retrieval.py`):
    -   The `retrieve_from_collection` function is called with the user's query.
    -   The query is embedded using the same model that was used for the documents.
    -   The embedded query is used to search the ChromaDB database for the most relevant text chunks.
    -   (Optional) The retrieved chunks are re-ranked using the `cross-encoder/ms-marco-MiniLM-L-6-v2` model to improve relevance.
3.  **Generation (`app.py`):
    -   The retrieved chunks and the user's query are passed to a Large Language Model (LLM).
    -   The LLM generates a comprehensive answer based on the provided information.
    -   The answer is then streamed back to the user in the web interface or printed in the CLI.

## Setup and Usage

### Prerequisites

- Python 3.x
- The required Python packages can be installed from the `requirements.txt` file:
  ```bash
  pip install -r requirements.txt
  ```

### Configuration

-   The main configuration is in the `config.yaml` file, which specifies the LLM model, API base, and other agent settings.
-   The prompts used by the agent are defined in the `prompts.yaml` file.
-   An `.env` file can be used to set environment variables, such as the `MODEL_LINK`.

### Running the Application

1.  **Data Ingestion:**
    -   To process new PDF files, run the `run_pipeline.py` script:
        ```bash
        python run_pipeline.py path/to/your/document.pdf
        ```
    -   If you run the script without any arguments, a file dialog will open to let you select the PDF files.

2.  **Running the Agent:**
    -   To run the web interface:
        ```bash
        python app.py
        ```
    -   The application will start a Gradio web server, and you can access the chat interface in your browser.

## Project Structure

```
f:\Spectroscopy Agent\
├───.gitignore
├───app.py                # Main application entry point (CLI and Gradio UI)
├───config.yaml           # Configuration file for the agent
├───Embedding.py          # Script for generating and storing embeddings
├───PDFParsing.py         # Script for parsing PDFs, extracting metadata, and chunking text
├───prompts.yaml          # Prompts for the LLM
├───requirements.txt      # Python dependencies
├───Retrieval.py          # Script for retrieving relevant chunks from the database
├───run_pipeline.py       # Orchestrates the data ingestion pipeline
├───SchemaHandler.py      # Manages the schema for tracking processed files
├───UI_Gradio.py          # Gradio user interface
├───.spectroscopy_chromadb\ # ChromaDB vector database
├───docs_pdfs\            # Directory for PDF documents
│   ├───papers\           # Renamed and processed papers
│   ├───Published\        # Input directory for published papers
│   └───Thesis\           # Input directory for theses
├───pdf_chunks_parquet\   # Directory for chunked text in Parquet format
└───toolbox\              # Directory for agent tools
    ├───final_answer.py
    ├───get_schema_info.py
    ├───list_collections.py # Unused tool.
    └───retrieve_chunks.py
```
