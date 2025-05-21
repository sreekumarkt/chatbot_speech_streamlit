import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize model
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech-to-text using microphone
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak into your mic.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand."
    except sr.RequestError:
        return "Speech service unavailable."

# Chatbot logic
def get_response(history, user_input):
    history.append(f"You: {user_input}")
    prompt = "\n".join(history) + "\nChatbot:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=inputs.shape[1] + 50, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    history.append(f"Chatbot: {reply}")
    return reply, history

# Streamlit UI
st.title("🗣️ Voice-Enabled Chatbot (Offline)")

chat_history = st.session_state.get("chat_history", [])

col1, col2 = st.columns([2, 1])
with col1:
    text_input = st.text_input("Type or click the mic to speak:")

with col2:
    if st.button("🎤 Speak"):
        text_input = listen()

if text_input:
    response, chat_history = get_response(chat_history, text_input)
    st.session_state.chat_history = chat_history
    st.markdown(f"**You:** {text_input}")
    st.markdown(f"**Chatbot:** {response}")
    speak(response)

