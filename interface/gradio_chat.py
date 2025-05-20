import gradio as gr
from app.chatbot import load_model, generate_response

tokenizer, model = load_model()

def chat(user_input, history):
    history = history or []
    response = generate_response(tokenizer, model, user_input, history)
    history.append((user_input, response))
    return history, history

with gr.Blocks(title="DeepSeek Chatbot") as demo:
    gr.Markdown("# 💬 DeepSeek Coder Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Digite sua mensagem")
    clear = gr.Button("Limpar")

    state = gr.State([])

    msg.submit(chat, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch()
