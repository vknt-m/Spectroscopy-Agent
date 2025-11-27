# ui_module.py
import gradio as gr
from typing import Any
import subprocess
import sys
from pathlib import Path

class GradioUI:
    """
    A modular Gradio UI for the Spectroscopy RAG Agent.
    This class encapsulates all UI logic and can be imported anywhere.
    """
    
    def __init__(self, agent):
        """
        Initialize the UI with an agent instance.
        
        Args:
            agent: The CodeAgent instance from Agent1.py
        """
        self.agent = agent
        
    # UI_Gradio.py - Modified chat_handler method
    def chat_handler(self, user_msg: str, history: list):
        """
        Handles user messages and sends them to the agent with streaming.
        Yields updates to the history for a real-time chat experience.
        """
        from smolagents.models import ChatMessageStreamDelta
        from smolagents.agents import FinalAnswerStep

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})

        try:
            # Enable streaming by setting stream=True
            response_generator = self.agent.run(user_msg, stream=True)
            
            # The content of the last message will be updated with the streamed response
            for step in response_generator:
                if isinstance(step, ChatMessageStreamDelta) and step.content:
                    history[-1]["content"] += step.content
                    yield history, "" # Yield the updated history to the UI
                elif isinstance(step, FinalAnswerStep):
                    # When the final answer is ready, we replace the streamed content with it
                    history[-1]["content"] = str(step.output)
                    yield history, ""

        except Exception as e:
            history[-1]["content"] = f"⚠️ Agent error: {str(e)}"
            yield history, ""

    
    def clear_memory(self):
        """
        Clears the agent's conversation memory.
        """
        self.agent.memory.steps = []
        return [], ""  # Clear chat history and input box

    def upload_file(self, file):
        """
        Handles file uploads and runs the ingestion pipeline.
        """
        if file is None:
            return "No file uploaded."

        file_path = Path(file.name)
        
        try:
            # Run the pipeline script as a subprocess
            # Using sys.executable ensures that we use the same python environment
            process = subprocess.run(
                [sys.executable, "run_pipeline.py", str(file_path)],
                capture_output=True,
                text=True,
                encoding="utf-8", 
                errors="replace",
                check=True
            )
            
            # If the script runs successfully, we'll get a success message.
            # You can customize this message as needed.
            output = f" **File '{file_path.name}' processed successfully!**\n\n"
            output += "You can now ask questions about its content.\n\n"
            output += "--- **Processing Details** ---\n"
            output += f"```\n{process.stdout}\n```"

        except subprocess.CalledProcessError as e:
            # If there's an error, we'll show the error message.
            output = f" **Error processing file '{file_path.name}'.**\n\n"
            output += "--- **Error Details** ---\n"
            output += f"```\n{e.stderr}\n```"
        
        except Exception as e:
            output = f"An unexpected error occurred: {str(e)}"

        return output
    
    def build_interface(self):
        """
        Builds and returns the Gradio interface.
        """
        with gr.Blocks(
            title="Spectroscopy RAG Agent Chat",
            theme="ocean",
            fill_height=True
        ) as interface:
            
            # Header
            gr.Markdown("## 🔬 Spectroscopy RAG Chatbot")
            gr.Markdown("Ask questions about your spectroscopy documents. The agent will retrieve relevant information and provide answers.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Main chat interface
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=500,
                        label="Conversation",
                        avatar_images=(None, None),  # No avatars
                        bubble_full_width=False
                    )
                    
                    # Input controls
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your question and press Enter...",
                            container=False,
                            scale=4
                        )
                        clear_btn = gr.Button("🔄 Clear Memory", scale=1)

                    #gr.Markdown("**Note:** Add `--deep` to your query to broaden the search range.")

                    
                    gr.Markdown("###  File Upload")
                    upload_button = gr.UploadButton("Click to Upload a PDF", file_types=[".pdf"])
                    upload_status = gr.Markdown(value="Upload a PDF to add it to the knowledge base.")

            
            # Event handlers
            msg_input.submit(
                fn=self.chat_handler,
                inputs=[msg_input, chatbot],  # Pass current chatbot state
                outputs=[chatbot, msg_input]  # Update chatbot and clear input
            )
            
            clear_btn.click(
                fn=self.clear_memory,
                outputs=[chatbot, msg_input]
            )

            upload_button.upload(
                fn=self.upload_file,
                inputs=[upload_button],
                outputs=[upload_status]
            )
            
        return interface
    
    def launch(self, **kwargs):
        """
        Builds and launches the Gradio interface.
        
        Args:
            **kwargs: Additional arguments passed to gradio.launch()
        """
        interface = self.build_interface()
        return interface.launch(**kwargs)

# Convenience function for quick setup
def create_ui(agent):
    """
    Factory function to create a UI instance.
    
    Args:
        agent: The CodeAgent instance
        
    Returns:
        SpectroscopyUI instance
    """
    return GradioUI(agent)
