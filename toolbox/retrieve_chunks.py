from smolagents import tool
from Retrieval import retrieve_from_collection
import json

def safe_text(s: str) -> str:
    return s.replace('"', "'").replace("\n", " ")

@tool
def retrieve_chunks(query: str, collection_name: str, n_results: int = 5, use_reranker: bool = False) -> str:
    """
    Retrieve the most relevant chunks from a specified collection.

    Args:
        query (str): The user's question to search for relevant chunks.
        collection_name (str): The name of the collection to search in.
        n_results (int): The number of top relevant chunks to return.
        use_reranker (bool): Optional: If True, uses a CrossEncoder model to re-rank
                               the results for improved relevance. Defaults to False.

    Returns:
        A JSON string representing a list of chunk objects, each with text and citation metadata.
    """
    
    results = retrieve_from_collection(
        query_text=query, 
        collection_name=collection_name, 
        n_results=n_results,
        use_reranker=use_reranker
    )

    if not results:
        return json.dumps([])

    # Sanitize results to convert non-serializable numpy types to standard python types.
    sanitized_results = []
    for res in results:
        sanitized_res = res.copy()
        if 'distance' in sanitized_res and sanitized_res['distance'] is not None:
            sanitized_res['distance'] = float(sanitized_res['distance'])
        if 'rerank_score' in sanitized_res and sanitized_res['rerank_score'] is not None:
            sanitized_res['rerank_score'] = float(sanitized_res['rerank_score'])
        sanitized_res['text'] = safe_text(sanitized_res.get('text', ''))
        sanitized_results.append(sanitized_res)

    return json.dumps(sanitized_results)
