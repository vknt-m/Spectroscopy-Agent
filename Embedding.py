import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import os
import sys # Import sys to read command line arguments
from SchemaHandler import SchemaHandler

# Configuration
PARQUET_PATH = Path("pdf_chunks_parquet")  # Directory of chunked Parquet files

DB_PATH = ".spectroscopy_chromadb"    # Directory to persist ChromaDB data
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Switched to a more powerful model
COLLECTION_NAME = "spectroscopy_books_papers"   # A constant holding the current single collection being used.


def main(collection_name: str = "spectroscopy_books_papers"):
    """
    Main function to load data in batches, generate embeddings, and ingest into ChromaDB
    in a memory-efficient and idempotent way, using SchemaHandler for tracking.
    """
    # --- 1. Initialize SchemaHandler ---
    schema_handler = SchemaHandler()

    # --- 2. Initialize ChromaDB Client and Embedding Function ---
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # --- 3. Get list of files to process from schema ---
    pending_files_df = schema_handler.get_pending_embeddings()
    if pending_files_df.empty:
        print("No new or updated files found in schema for embedding. Database is up-to-date.")
        return

    print(f"Found {len(pending_files_df)} files in schema pending embedding. Ingesting into '{collection_name}'...")

    total_chunks_ingested = 0
    for index, row in tqdm(pending_files_df.iterrows(), total=len(pending_files_df), desc="Embedding files"):
        filename = row['filename']
        parquet_file_path = PARQUET_PATH / f'{Path(filename).stem}.parquet'

        if not parquet_file_path.exists():
            print(f"Warning: Parquet file for '{filename}' not found at '{parquet_file_path}'. Skipping.")
            schema_handler.update_entry({'filename': filename}, status="parquet_missing")
            continue

        try:
            df = pd.read_parquet(parquet_file_path)
            if df.empty:
                print(f"Warning: Parquet file for '{filename}' is empty. Skipping.")
                schema_handler.update_entry({'filename': filename}, status="empty_parquet")
                continue

            # --- Idempotency Check ---
            # Generate potential IDs for all chunks in the current file
            potential_ids = [f"{row_df['source_filename']}_{row_df['page_number']}_{i}" for i, row_df in df.iterrows()]

            # Check which of these IDs already exist in the database
            existing_ids_response = collection.get(ids=potential_ids, include=[])
            existing_ids = set(existing_ids_response['ids'])

            # Filter the DataFrame to only include new, non-existent chunks
            if existing_ids:
                df['potential_id'] = potential_ids
                new_chunks_df = df[~df['potential_id'].isin(existing_ids)]
            else:
                new_chunks_df = df

            if new_chunks_df.empty:
                print(f"No new chunks to ingest for '{filename}'. Marking as embedded.")
                schema_handler.mark_embedded(filename) # Mark as embedded even if no new chunks
                continue

            # --- Process and Ingest only the New Chunks ---
            # Ensure all metadata and text columns are strings
            for col in ["chunk_text", "source_filename", "title", "author", "year", "page_number"]:
                if col in new_chunks_df.columns:
                    new_chunks_df.loc[:, col] = new_chunks_df[col].astype(str).fillna("")

            documents = new_chunks_df["chunk_text"].tolist()
            metadatas = new_chunks_df[["source_filename", "title", "author", "year", "page_number"]].to_dict('records')
            
            # Use the pre-generated potential_ids that correspond to the new chunks
            ids_to_add = new_chunks_df['potential_id'].tolist() if existing_ids else potential_ids

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids_to_add
            )
            total_chunks_ingested += len(new_chunks_df)
            schema_handler.mark_embedded(filename) # Mark as embedded after successful ingestion

        except Exception as e:
            print(f"\nError processing file '{filename}': {e}")
            schema_handler.update_entry({'filename': filename}, status="embedding_failed")
            continue

    print("\n--- Ingestion Complete! ---")
    if total_chunks_ingested > 0:
        print(f"Successfully ingested {total_chunks_ingested} new chunks into the database.")
    else:
        print("No new chunks found to ingest. The database is already up-to-date.")
        
    print(f"Total chunks in collection '{collection_name}': {collection.count()}")
    print(f"Your vector database is persisted in the '{DB_PATH}' directory.")

    # Save the updated schema
    schema_handler.save()

if __name__ == "__main__":
    # Allow collection name to be passed as a command-line argument
    # Example: python Embedding.py my_new_collection
    if len(sys.argv) > 1:
        main(collection_name=sys.argv[1])
    else:
        main() # Use default collection name
