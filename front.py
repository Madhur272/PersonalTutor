import streamlit as st
import requests
import os

st.set_page_config(page_title="NLP Personal Tutor", page_icon="🤖")

BACKEND_DEFAULT = "http://localhost:8000"

backend_url = st.sidebar.text_input("Backend URL", value=BACKEND_DEFAULT)

st.title("Personal Tutor ChatBot 🤖")
st.write("Solves all your academic queries.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Clear chat"):
        st.session_state.messages = []


# small helper to render messages
def render_messages():
    for m in st.session_state.messages:
        role = m.get("role", "user")
        text = m.get("text", "")
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")


def _post_to_backend(user_text: str):
    payload = {
        "query": user_text,
    }
    # include server-side session id if we have one
    if st.session_state.get("session_id"):
        payload["session_id"] = st.session_state.get("session_id")
    # include chat history for context fallback
    payload["chat_history"] = [{"role": m["role"], "text": m["text"]} for m in st.session_state.messages]
    try:
        resp = requests.post(f"{backend_url}/chat", json=payload, timeout=60)
        try:
            return resp.json()
        except Exception:
            return {"answer": resp.text}
    except Exception as e:
        return {"answer": f"Error contacting backend: {e}"}


# Callback used by the form submit button. Runs before rerun, so it's
# safe to mutate st.session_state inside this function.
def send_message():
    user_text = st.session_state.get("input_box", "").strip()
    if not user_text:
        return
    # append user message immediately
    st.session_state.messages.append({"role": "user", "text": user_text})
    with st.spinner("Waiting for answer..."):
        data = _post_to_backend(user_text)
    answer = data.get("answer") if isinstance(data, dict) else str(data)
    # if backend returned a session_id, store it for future calls
    if isinstance(data, dict) and data.get("session_id"):
        st.session_state.session_id = data.get("session_id")
    st.session_state.messages.append({"role": "bot", "text": answer})
    # clear input box safely
    st.session_state["input_box"] = ""


# Input form
with st.form(key="input_form"):
    st.text_input("Your message:", key="input_box")
    st.form_submit_button("Send", on_click=send_message)


# Render conversation
render_messages()

st.markdown("---")
st.write("Tip: run `python app.py` to start the backend, then `streamlit run front.py` to open this UI.")
