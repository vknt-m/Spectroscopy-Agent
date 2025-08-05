# ui_module.py
import gradio as gr
from typing import Any

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
        Handles user messages and sends them to the agent.
        Returns messages in the proper format for type="messages" chatbot.
        """
        try:
            response = self.agent.run(user_msg)

            # Add user message
            history.append({
                "role": "user", 
                "content": user_msg
            })

            # Add assistant response
            history.append({
                "role": "assistant", 
                "content": str(response)
            })

            return history, ""  # Clear input box after submission

        except Exception as e:
            # Add user message
            history.append({
                "role": "user", 
                "content": user_msg
            })

            # Add error message as assistant response
            history.append({
                "role": "assistant", 
                "content": f"⚠️ Agent error: {str(e)}"
            })

            return history, ""

    
    def clear_memory(self):
        """
        Clears the agent's conversation memory.
        """
        self.agent.memory.steps = []
        return [], ""  # Clear chat history and input box
    
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
            
            # Example questions
            gr.Examples(
                examples=[
                    "What is Raman spectroscopy?",
                    "Explain the principles of fluorescence lifetime imaging.",
                    "How does Fourier-transform infrared spectroscopy work?",
                    "What are the applications of mass spectrometry?"
                ],
                inputs=msg_input
            )
            
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
