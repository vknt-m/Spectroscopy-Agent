import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
DB_PATH = ".spectroscopy_chromadb"
#COLLECTION_NAME = "spectroscopy_books_papers"
# IMPORTANT: This must be the same model you used for embedding the documents
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def retrieve_from_collection(query_text: str, collection_name: str, n_results: int = 3):
    """
    Queries a SPECIFIC ChromaDB collection to find the most relevant chunks.

    Args:
        query_text (str): The user's question or search term.
        collection_name (str): The name of the collection to query.
        n_results (int): The number of relevant chunks to retrieve.

    Returns:
        A list of dictionaries containing the retrieved chunks and their metadata.
    """
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    except ValueError:
        print(f"[ERROR] Collection '{collection_name}' not found.")
        return []

    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"] 
    )

    

    if not results or not results.get('ids')[0]:
        print("--- No relevant documents found. ---")
        return []
        
    retrieved_chunks = []
    print(f"--- Found {len(results['ids'][0])} relevant chunks ---\n")
    
    for i in range(len(results['ids'][0])):
        chunk_info = {
            
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i].get('sourcefilename', 'N/A'),
            'page': results['metadatas'][0][i].get('pagenumber', 'N/A'),
            'distance': round(results['distances'][0][i], 4)
        }
        retrieved_chunks.append(chunk_info)
        
    return retrieved_chunks


