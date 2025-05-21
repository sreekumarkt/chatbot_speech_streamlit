import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tempfile
import base64

# Cache the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text-to-speech using gTTS
def speak(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(
        f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )
    os.remove(audio_path)

# Speech-to-text using microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Generate chatbot response
def get_response(history, user_input):
    history.append(f"User: {user_input}")
    prompt = "\n".join(history) + "\nBot:"

    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_k=50, top_p=0.95)

    reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()
    history.append(f"Bot: {reply}")
    return reply, history[-6:]  # keep last 3 turns

# UI
st.title("🗣️ Voice-Enabled Chatbot (Offline/Cloud-Compatible)")
chat_history = st.session_state.get("chat_history", [])

col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_input("Type a message or use the mic:", key="input")
with col2:
    if st.button("🎤 Speak"):
        text_input = listen()
        st.session_state.input = text_input

if text_input:
    response, chat_history = get_response(chat_history, text_input)
    st.session_state.chat_history = chat_history

    st.markdown(f"**You:** {text_input}")
    st.markdown(f"**Bot:** {response}")
    speak(response)

# Show full history
with st.expander("💬 Conversation History"):
    for line in chat_history:
        st.markdown(line)

