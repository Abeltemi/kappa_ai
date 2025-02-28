import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv(".env")

openai = OpenAI()

character_one = """
You are a helpful assistant that teaches French language.
"""

character_two = """
You are the best comedian in the world. But you speak and crack jokes both in Nigerian Pidgin language and in English.
"""

MODEL = "gpt-4o-mini"

def chat_model(messages, stream=False):
    if stream:
        chunks = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )
        return chunks
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response

def chat(message, history, model_character):
    messages = []
    if model_character == "French Teacher":
        messages = [{"role": "system", "content": character_one}]
    elif model_character == "Comedian":
        messages = [{"role": "system", "content": character_two}]
    

    history = history or []
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})

    chunks = chat_model(messages, stream=True)
    responses = ""
    for chunk in chunks:
        delta_content = chunk.choices[0].delta.content or ""
        responses += delta_content
        if responses: 
            yield responses

def transcribe_audio(audio_file):
    if audio_file is None:
        return ""
    try:
        with open(audio_file, "rb") as audio:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
            )
        return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# gradio interace design
with gr.Blocks() as b:
    gr.Markdown("# Choose a Preferred Character")
    character_dropdown = gr.Dropdown(
        choices=["French Teacher", "Comedian"],
        label="Select Character",
        value="French Teacher"
    )
    chatbot = gr.Chatbot(label="Kappa_ai")
    with gr.Row():
        textbox = gr.Textbox(label="Enter Text...")
        audio_mic = gr.Audio(sources="microphone", type="filepath", label="Record Audio")
    btn = gr.Button("Send")
    
    def submit_message(message, history, character):
        if not message:
            return history or [], "" 
        
        history = history or []
        history.append([message, None])
        for response in chat(message, history[:-1], character):
            history[-1][1] = response
            yield history, ""  

    def submit_audio(audio, history, character):
        transcribed_text = transcribe_audio(audio)
        if not transcribed_text or "Error" in transcribed_text:
            return history or [], transcribed_text
        
        history = history or []
        history.append([transcribed_text, None])
        for response in chat(transcribed_text, history[:-1], character):
            history[-1][1] = response
            yield history, transcribed_text 

    textbox.submit(
        fn=submit_message,
        inputs=[textbox, chatbot, character_dropdown],
        outputs=[chatbot, textbox]
    )
    btn.click(
        fn=submit_audio,
        inputs=[audio_mic, chatbot, character_dropdown],
        outputs=[chatbot, textbox] 
    )


def main():
    b.queue().launch(inbrowser=True, share=True)
    
if __name__ == '__main__':
    main()