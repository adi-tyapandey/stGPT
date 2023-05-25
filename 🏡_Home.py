import streamlit as st
from streamlit_extras.colored_header import colored_header

st.set_page_config(
    page_title="PDFGPT",
    page_icon="🤖",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

colored_header(
    label="Interactive AI Notebook 📚",
    description="Navigate to 🗃️Upload PDF (to upload your PDF) || Navigate to 📕PDF Chat (to talk to your PDF)",
    color_name="red-70",
    )

st.markdown('''\
        Dive into your instruction manuals, textbooks, storybooks and any other document using this webapp!
        
        Have questions or need assistance with your document? No worries! Simply upload you document and ask a question to get tailored, relevant answers.
        It uses large language models and conversational retrieval techniques (Embeddings) to provide you with the most accurate responses.
        
        ### ✨Features✨
        
        - ✅ Choose from a list of available PDF options or upload your own for exploration.
        - ✅ Engage in a chat interface to inquire about the selected manual.
        - ✅ Experience an intuitive display of answers along with the related source documents.
        - ⚒️ **Coming Soon:** Autonomous AI agent with tools and Internet access that can help you complete tasks.
        
        ### 🚀 Technologies Used 🚀
        
        - ✅ Python
        - ✅ Langchain library
        - ✅ LLM
        - ✅ Streamlit framework UI
        
        Get ready to unlock the power of interactive PDF exploration! 📚💻💬
    ''')
st.write("[Github](https://github.com/adi-tyapandey)")
