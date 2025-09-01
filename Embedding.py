import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Configuration
PARQUET_PATH = "pdf_chunks.parquet"  # Directory of chunked Parquet files
DB_PATH = ".spectroscopy_chromadb"    # Directory to persist ChromaDB data
COLLECTION_NAME = "spectroscopy_books_papers"  # Name of the collection in ChromaDB
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Switched to a more powerful model


def main():
    """
    Main function to load data in batches, generate embeddings, and ingest into ChromaDB
    in a memory-efficient and idempotent way.
    """
    # --- 1. Initialize ChromaDB Client and Embedding Function ---
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # --- 2. Get list of files to process ---
    try:
        parquet_files = [os.path.join(PARQUET_PATH, f) for f in os.listdir(PARQUET_PATH) if f.endswith('.parquet')]
        if not parquet_files:
            print(f"No .parquet files found in '{PARQUET_PATH}'.")
            print("Please ensure you have run the parsing and chunking script first.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{PARQUET_PATH}' was not found.")
        print("Please ensure you have run the parsing and chunking script first.")
        return

    # --- 3. Process and Ingest Files in Batches ---
    total_chunks_ingested = 0
    print(f"Found {len(parquet_files)} files. Checking for new content to ingest...")

    for file_path in tqdm(parquet_files, desc="Ingesting files"):
        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue

            # --- Idempotency Check ---
            # 1. Generate potential IDs for all chunks in the current file
            potential_ids = [f"{row['source_filename']}_{row['page_number']}_{i}" for i, row in df.iterrows()]

            # 2. Check which of these IDs already exist in the database
            existing_ids_response = collection.get(ids=potential_ids, include=[])  # Only need IDs, no embeddings or metadata
            existing_ids = set(existing_ids_response['ids'])

            # 3. Filter the DataFrame to only include new, non-existent chunks
            if existing_ids:
                df['potential_id'] = potential_ids
                new_chunks_df = df[~df['potential_id'].isin(existing_ids)]
            else:
                new_chunks_df = df # If no IDs existed, the whole dataframe is new

            # 4. If there are no new chunks in this file, skip to the next one
            if new_chunks_df.empty:
                continue

            # --- Process and Ingest only the New Chunks ---
            # Ensure all metadata and text columns are strings
            for col in ["chunk_text", "source_filename", "title", "author", "year", "page_number"]:
                if col in new_chunks_df.columns:
                    new_chunks_df.loc[:, col] = new_chunks_df[col].astype(str).fillna("")

            documents = new_chunks_df["chunk_text"].tolist()
            metadatas = new_chunks_df[["source_filename", "title", "author", "year", "page_number"]].to_dict('records')
            
            # Use the pre-generated potential_ids that correspond to the new chunks
            if existing_ids:
                ids_to_add = new_chunks_df['potential_id'].tolist()
            else:
                ids_to_add = potential_ids

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids_to_add
            )
            total_chunks_ingested += len(new_chunks_df)

        except Exception as e:
            print(f"\nError processing file {os.path.basename(file_path)}: {e}")
            print("Skipping this file.")
            continue

    print("\n--- Ingestion Complete! ---")
    if total_chunks_ingested > 0:
        print(f"Successfully ingested {total_chunks_ingested} new chunks into the database.")
    else:
        print("No new chunks found to ingest. The database is already up-to-date.")
        
    print(f"Total chunks in collection '{COLLECTION_NAME}': {collection.count()}")
    print(f"Your vector database is persisted in the '{DB_PATH}' directory.")


if __name__ == "__main__":
    main()
