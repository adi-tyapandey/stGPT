import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from gpt4free import usesless

def aiCompletion(question: str) -> str:
    context = "\n".join([chat['message'] for chat in st.session_state.history])
    prompt_with_context = f"{context}\n{question}"
    req = usesless.Completion.create(prompt=prompt_with_context, parentMessageId="")
    st.session_state.history.append({"message": question, "is_user": True})
    st.session_state.history.append({"message": req['text'], "is_user": False})
    return req['text']

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Chat", page_icon="🤖")
colored_header(
    label="Chat Bot 👱🏽",
    description="Optimized for conversation",
    color_name="red-70",
)

with st.form("question_form"):
  col1, col2 = st.columns([5, 1])
  with col1:
    user_question = st.text_input('')
  with col2:
    st.write('')
    st.write('')
    submit_button = st.form_submit_button('🧠 Think')
if user_question and submit_button:
  with st.spinner('Loading...'):
    try:
      answer = aiCompletion(user_question)
    except:
      st.session_state.history = []
      st.error('Request Failed. Please try again')
    for i, chat in enumerate(st.session_state.history[::-1]):
      message(**chat, key=str(i))
