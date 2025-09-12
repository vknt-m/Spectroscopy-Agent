from smolagents import tool
from Retrieval import retrieve_from_collection
import json # To handle the metadata_filter string

@tool
def retrieve_chunks(query: str, collection_name: str, n_results: int = 5, metadata_filter: str = None, use_reranker: bool = False) -> str:
    """
    Retrieve the most relevant chunks from a specified collection.

    Args:
        query (str): The user's question to search for relevant chunks.
        collection_name (str): The name of the collection to search in.
        n_results (int): The number of top relevant chunks to return.
        metadata_filter (str): Optional: A JSON string for metadata filtering. 
                               Example: '{"author": "John Doe", "year": "2023"}'
        use_reranker (bool): Optional: If True, uses a CrossEncoder model to re-rank
                               the results for improved relevance. Defaults to False.

    Returns:
        A formatted string with chunk text and citation metadata.
    """
    
    parsed_metadata_filter = None
    if metadata_filter:
        try:
            parsed_metadata_filter = json.loads(metadata_filter)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for metadata_filter."

    results = retrieve_from_collection(
        query_text=query, 
        collection_name=collection_name, 
        n_results=n_results,
        metadata_filter=parsed_metadata_filter,
        use_reranker=use_reranker
    )

    if not results:
        return "No relevant chunks found."

    out = []
    for res in results:
        rerank_info = f" (Re-rank Score: {res['rerank_score']:.4f})" if 'rerank_score' in res else ""
        distance_info = f" (Distance: {res['distance']:.4f})"
        out.append(f"[{res['id']}] (Source: {res['source']}, Page: {res['page']}){distance_info}{rerank_info}\nText: {res['text'][:350]}...")
    return "\n\n".join(out)