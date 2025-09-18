from dotenv import load_dotenv
from poml.integration.langchain import LangchainPomlTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
import gradio as gr

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

conversation_history = []

conversation_history = []

def get_response(user_input):
    global conversation_history
    prompt = LangchainPomlTemplate.from_file("prompt.poml")
    history_text = "\n".join([f"Human: {h['user']}\nAssistant: {h['bot']}" 
                              for h in conversation_history[-5:]])
    context = {"question": user_input, "history": history_text}
    chain = prompt | llm | StrOutputParser()
    ai_response = chain.invoke(context)
    conversation_history.append({"user": user_input, "bot": ai_response})
    if len(conversation_history) > 10:
        conversation_history.pop(0)
    return ai_response

PAGE_TITLE = "Caramel AI from HERE AND NOW AI"
LOGO_URL = "https://raw.githubusercontent.com/hereandnowai/images/refs/heads/main/logos/logo-of-here-and-now-ai.png"
ASSISTANT_AVATAR_URL = "https://raw.githubusercontent.com/hereandnowai/images/refs/heads/main/logos/caramel-face.jpeg"

description_markdown = f"""
<img src='{LOGO_URL}' width='500' style='display: block; margin: auto;'>
<br>Your friendly AI teacher for learning the basics of Artificial Intelligence
"""

# Example questions for AI beginners
example_questions = [
    "What is Artificial Intelligence?",
    "What is machine learning?",
    "Difference between AI and ML?",
    "Can you explain neural networks in simple words?",
    "What are the main applications of AI?"
]

def respond(message, chat_history):
    response = get_response(message)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})    
    return chat_history, ""  # Clear input box

# Build Gradio interface
with gr.Blocks(title=PAGE_TITLE) as demo:
    gr.Markdown(description_markdown)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(type="messages", avatar_images=[None, ASSISTANT_AVATAR_URL])
            user_input = gr.Textbox(placeholder="Type your message here...")
            submit_btn = gr.Button("Send")
        with gr.Column(scale=1):
            gr.Markdown("### Example Questions")
            for q in example_questions:
                gr.Button(q, elem_id=q).click(
                     lambda x=q: respond(x, chatbot_ui.value),  # Pass current chat history
                    inputs=[],
                    outputs=[chatbot_ui, user_input]
                )
    
    submit_btn.click(respond, inputs=[user_input, chatbot_ui], outputs=[chatbot_ui, user_input])  
    user_input.submit(respond, inputs=[user_input, chatbot_ui], outputs=[chatbot_ui, user_input])
demo.launch(share=True)



if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = get_response(user_input)
        print(f"Bot: {response}")