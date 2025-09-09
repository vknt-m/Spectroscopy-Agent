from smolagents import tool
import chromadb

# --- Configuration ---
DB_PATH = ".spectroscopy_chromadb"

@tool
def list_collections() -> str:
    """
    Lists all available document collections in the ChromaDB database.

    Returns:
        A comma-separated string of available collection names, or a message if none are found.
    """
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collections = client.list_collections()
        if not collections:
            return "No collections found."
        
        # Extract names from the Collection objects
        collection_names = [collection.name for collection in collections]
        return ", ".join(str(collection_names))
    except Exception as e:
        return f"An error occurred while listing collections: {e}"