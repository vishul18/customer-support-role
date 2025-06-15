import openai
import base64
from io import BytesIO
from PIL import Image
from IPython.display import Audio, display
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"
system_message = "You are a helpful assistant."
tools = []

def artist(city):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=f"A vacation in {city}, vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    img_data = base64.b64decode(response.data[0].b64_json)
    return Image.open(BytesIO(img_data))

def talker(message):
    response = openai.audio.speech.create(model="tts-1", voice="onyx", input=message)
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    display(Audio("output.mp3", autoplay=True))

def chat(history):
    messages = [{"role": "system", "content": system_message}] + history
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]
    talker(reply)
    return history, None

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500)
    entry = gr.Textbox(label="Chat with our AI Assistant:")
    clear = gr.Button("Clear")

    def on_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    entry.submit(on_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]) \
         .then(chat, inputs=chatbot, outputs=[chatbot, image_output])

    clear.click(lambda: ([], None), outputs=[chatbot, image_output], queue=False)

ui.launch(inbrowser=True)
