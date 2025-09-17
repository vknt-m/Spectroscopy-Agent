import os
from dotenv import load_dotenv
from smolagents import CodeAgent,DuckDuckGoSearchTool, load_tool,LiteLLMModel #HfApiModel, OpenAIServerModel
import requests
import yaml
import sys
from UI_Gradio import GradioUI as create_ui  # Import the Gradio UI class
from toolbox.final_answer import FinalAnswerTool
from toolbox.retrieve_chunks import retrieve_chunks
#from toolbox.create_metadata_filter import create_metadata_filter
from toolbox.list_collections import list_collections
from toolbox.list_items_from_collection import list_items_from_collection

# Load environment variables from .env file
load_dotenv()

# Load configuration
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

final_answer = FinalAnswerTool()

# Determine the API base: use MODEL_LINK from .env if available, otherwise use api_base from config.yaml
api_base = os.getenv("MODEL_LINK") or config['llm']['api_base']

llm = LiteLLMModel(
    model_id=config['llm']['model_id'],
    api_base=api_base,
    num_ctx=config['llm']['num_ctx'],
)

# Register your tools
tools = [retrieve_chunks, final_answer, list_collections, list_items_from_collection]

# Load prompts
with open(config['prompts_file'], 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# Initialize the agent with your LLM and tools
agent = CodeAgent(
    model=llm,
    tools=tools,
    prompt_templates=prompt_templates,
    verbosity_level=config['agent']['verbosity_level'],
    max_steps=config['agent']['max_steps'],
    additional_authorized_imports=['json'],
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
    
    run_ui()
    #if len(sys.argv) > 1 and sys.argv[1] == "--ui":
    #    run_ui()
    #else:
    #run_cli()
