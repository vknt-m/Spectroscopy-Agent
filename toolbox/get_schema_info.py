import pandas as pd
import json
from pathlib import Path
from smolagents.tools import tool

schema_path = "info_schema.json"

@tool
def get_schema_info() -> str:
    """
    Loads the schema file and returns it as a JSON string.
    Useful for letting the agent inspect available PDFs,
    their metadata, and processing statuses.
    
    Returns:
        str: A JSON string containing the schema contents or an error message.
    """
    schema_file = Path(schema_path)

    if not schema_file.exists():
        return json.dumps({"error": f"Schema file not found at: {schema_path}"})


    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Ensure it is always a list of dicts (schema format)
        if isinstance(data, dict):
            data = [data]

        # Safely convert DataFrame to JSON for the agent
        df = pd.DataFrame(data)
        return df.to_json(orient="records", force_ascii=False)

    except json.JSONDecodeError:
        return json.dumps({"error": f"Could not decode JSON from {schema_path}"})

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print(get_schema_info())
