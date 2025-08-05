# %%
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
CSV_PATH = "spectroscopy_chunks.csv"  # Your chunked CSV with columns: chunktext, sourcefilename, title, author, year, pagenumber
DB_PATH = ".spectroscopy_chromadb"    # Directory to persist ChromaDB data
COLLECTION_NAME = "spectroscopy_books_papers"  # Name of the collection in ChromaDB
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Local sentence-transformer model



# %%

def main():
    """
    Main function to load data, generate embeddings, and ingest into ChromaDB.
    """
    # --- 1. Load the Processed Chunks ---
    try:
        df = pd.read_csv(CSV_PATH)
        # Ensure all metadata and text columns are strings to prevent errors
        for col in ["chunk_text", "source_filename", "title", "author", "year", "page_number"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
        
        print(f"Successfully loaded {len(df)} chunks from '{CSV_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_PATH}' was not found.")
        print("Please ensure you have run the parsing and chunking script first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return
    

    # --- 2. Initialize ChromaDB Client and Embedding Function ---
    
    # Set up the ChromaDB client with persistence. This saves the database locally.
    client = chromadb.PersistentClient(path=DB_PATH)

    # Set up the embedding function using the corrected 'embedding_functions' module.
    # This helper from ChromaDB will correctly use the 'sentence_transformers' library
    # in the background to create embeddings.
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # --- 3. Create or Get the ChromaDB Collection ---
    # The embedding function is passed at creation time. ChromaDB will use it automatically.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # --- 4. Prepare and Ingest Data into ChromaDB ---
    documents = df["chunk_text"].tolist()
    # Ensure all metadata is properly formatted as a list of dictionaries
    metadatas = df[["source_filename", "title", "author", "year", "page_number"]].to_dict('records')
    # Create a unique and descriptive ID for each chunk
    ids = [f"chunk_{i}_{row['source_filename']}" for i, row in df.iterrows()]

    print(f"Starting ingestion of {len(documents)} chunks into the '{COLLECTION_NAME}' collection...")
    
    # Add the data to the collection.
    # ChromaDB handles batching and embedding automatically.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("\n--- Ingestion Complete! ---")
    print(f"Successfully added {collection.count()} chunks to the ChromaDB collection.")
    print(f"Your vector database is persisted in the '{DB_PATH}' directory.")


if __name__ == "__main__":
    main()


# %%



