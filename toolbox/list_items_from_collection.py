from smolagents import tool
import chromadb

# --- Configuration ---
DB_PATH = ".spectroscopy_chromadb"

@tool
def list_items_from_collection(collection_name: str, n_items: int = 5) -> str:
    """
    Lists a specified number of items from a single collection in the ChromaDB database.

    Args:
        collection_name (str): The name of the collection to list items from.
        n_items (int): The number of items to list. Defaults to 5.

    Returns:
        A formatted string with the item IDs and their text, or an error message.
    """
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=collection_name)
        
        results = collection.get(limit=n_items)

        if not results or not results['ids']:
            return f"No items found in collection '{collection_name}'."

        out = []
        for i, item_id in enumerate(results['ids']):
            text = results['documents'][i]
            out.append(f"[{item_id}] Text: {text[:350]}...")
        return "\n\n".join(out)

    except Exception as e:
        return f"An error occurred while listing items from collection '{collection_name}': {e}"