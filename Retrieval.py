import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder # Added for re-ranking

# --- Configuration ---
DB_PATH = ".spectroscopy_chromadb"
#COLLECTION_NAME = "spectroscopy_books_papers"
# IMPORTANT: This must be the same model you used for embedding the documents
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# --- Initialize Re-ranking Model (loaded once) ---
# This model will be used to re-rank the search results for better relevance.
RERANKER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def retrieve_from_collection(query_text: str, collection_name: str, n_results: int = 3, metadata_filter: dict = None, use_reranker: bool = False):
    """
    Queries a SPECIFIC ChromaDB collection to find the most relevant chunks.

    Args:
        query_text (str): The user's question or search term.
        collection_name (str): The name of the collection to query.
        n_results (int): The number of relevant chunks to retrieve.
        metadata_filter (dict, optional): A dictionary for metadata filtering. 
                                          Example: {"author": "John Doe", "year": "2023"}
        use_reranker (bool, optional): If True, uses a CrossEncoder model to re-rank
                                       the results for improved relevance. Defaults to False.

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

    # Add the instruction to the query for optimal performance with BGE models
    query_with_instruction = f"Represent this sentence for searching relevant passages: {query_text}"

    # If re-ranking, fetch more initial candidates to give the re-ranker a better pool to choose from.
    num_candidates = n_results * 5 if use_reranker else n_results

    results = collection.query(
        query_texts=[query_with_instruction],
        n_results=num_candidates,
        where=metadata_filter,
        include=["documents", "metadatas", "distances"]
    )

    

    if not results or not results.get('ids')[0]:
        print("--- No relevant documents found. ---")
        return []
        
    retrieved_chunks = []
    print(f"--- Found {len(results['ids'][0])} relevant chunks ---")
    
    for i in range(len(results['ids'][0])):
        chunk_info = {
            
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i].get('sourcefilename', 'N/A'),
            'page': results['metadatas'][0][i].get('pagenumber', 'N/A'),
            'distance': round(results['distances'][0][i], 4)
        }
        retrieved_chunks.append(chunk_info)
        
    # --- Optional Re-ranking Step ---
    if use_reranker and retrieved_chunks:
        print(f"--- Re-ranking top {len(retrieved_chunks)} results... ---")
        
        # Create pairs of [query, document_text] for the cross-encoder
        pairs = [[query_text, chunk['text']] for chunk in retrieved_chunks]
        
        # Predict the relevance scores
        scores = RERANKER_MODEL.predict(pairs, show_progress_bar=False)

        # Add the re-rank score to each chunk
        for i in range(len(retrieved_chunks)):
            retrieved_chunks[i]['rerank_score'] = scores[i]

        # Sort the chunks by the new re-rank score in descending order
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x['rerank_score'], reverse=True)

        # Truncate the list to the original n_results
        retrieved_chunks = retrieved_chunks[:n_results]

    return retrieved_chunks



