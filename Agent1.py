from smolagents import CodeAgent,DuckDuckGoSearchTool, load_tool,tool,LiteLLMModel #HfApiModel, OpenAIServerModel 
import requests
import yaml
from secretstuff import model_link , TOGETHER_API_KEY # Import your secrets file for API keys and URLs , optional but recommended.

from UI_Gradio import GradioUI as create_ui  # Import the Gradio UI class
from finalanswer import FinalAnswerTool


final_answer = FinalAnswerTool()

@tool
def retrieve_chunks(query: str, n_results: int = 5) -> str:
    """
    Retrieve the most relevant chunks from the spectroscopy_books_papers collection.

    Args:
        query: The user's question to search for relevant chunks.
        n_results: The number of top relevant chunks to return.

    Returns:
        A formatted string with chunk text and citation metadata.
    """
    # ... function body ...

    from Retrieval import retrieve_from_collection  #retrieval logic
    results = retrieve_from_collection(query, collection_name="spectroscopy_books_papers", n_results=n_results)
    if not results:
        return "No relevant chunks found."
    out = []
    for res in results:
        out.append(f"[{res['id']}] (Page {res['page']}) {res['text'][:350]}...")
    return "\n\n".join(out)



llm = LiteLLMModel(
    model_id="ollama_chat/gemma3:4b",  # e.g., "gpt-3.5-turbo" or your local model's name/alias
    api_base= model_link,  # ngrok or local URL for your LLM server
    #api_key="sk-..."  # Use a dummy value if your local LLM does not require a key
    num_ctx=16384,  # Adjust based on your model's context length
    
)



# Register your tools
tools1 = [retrieve_chunks,final_answer]

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream) 
#with open("prompts_md_4ui.yaml", 'r') as stream:
#    prompt_templates = yaml.safe_load(stream) 


# Initialize the agent with your LLM and tools
agent = CodeAgent(
    model=llm,
    tools=tools1,
    prompt_templates=prompt_templates,      
    verbosity_level=2,  # Set to 2 for detailed output
    max_steps=6,  # Limit the number of iterations to prevent infinite loops             
)

def run_cli():
    """Run the command-line interface."""
    print("Spectroscopy Research Agent is ready. Type your question or 'exit' to quit.\n")
    while True:
        user_query = input("> ")
        if user_query.strip().lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            response = agent.run(user_query)
            print("\nAssistant:\n" + str(response) + "\n" + "-"*50)
        except Exception as e:
            print(f"Error: {e}")

def run_ui():
    """Run the Gradio web interface."""
    ui = create_ui(agent)
    ui.launch(debug=True, share=True)

if __name__ == "__main__":
    import sys
    run_ui()
    #if len(sys.argv) > 1 and sys.argv[1] == "--ui":
    #    run_ui()
    #else:
    #    run_cli()


