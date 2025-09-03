from smolagents import tool
import json

@tool
def create_metadata_filter(author: str = None, year: int = None) -> str:
    """
    Creates a JSON string for filtering documents by metadata.

    Args:
        author (str, optional): The author to filter by.
        year (int, optional): The publication year to filter by.

    Returns:
        A JSON string representing the metadata filter, or an empty string if no filters are provided.
    """
    metadata_filter = {}
    if author:
        metadata_filter['author'] = author
    if year:
        metadata_filter['year'] = year
    
    if not metadata_filter:
        return ""
        
    return json.dumps(metadata_filter)