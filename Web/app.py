import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

from engine import process_pdf, create_conversation_chain, load_model_embedding, load_model_chat

st.set_page_config(page_title="chat com seu PDF", layout='wide')
st.title("Muilti PDF Chat")

@st.cache_resource
def get_embedding_model():
    return load_model_embedding()

@st.cache_resource
def get_llm_model():
    return load_model_chat()

model_embedding = get_embedding_model()
model_llm = get_llm_model()


with st.sidebar:
    st.header("Load docs")
    loaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if loaded_file:
        if st.button("Process document"):
            with st.spinner("Document processed, please wait..."):
                
                vector_db = process_pdf(loaded_file, model_embedding)
                if vector_db is None:
                    st.error("Vector database creation failed. Please check your PDF or embeddings.")
                    st.stop()

                st.session_state.vector_db = vector_db
                st.session_state.chat_memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="answer"
                )
                st.session_state.conversation_chain = create_conversation_chain(
                    st.session_state.vector_db,
                    model_llm,
                    st.session_state.chat_memory
                )

                st.session_state.messages = []
                st.success("Document processed successfully! You can now ask questions about it.")

st.header("Ask questions about your document")

if "conversation_chain" in st.session_state:
    
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_question = st.chat_input("Type your question here...")
    if user_question:
        st.session_state.messages.append({"role":"user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Thinking..."):
            result = st.session_state.conversation_chain.invoke({"question": user_question})
            print(result)
            answer_ia = result["answer"]

            st.session_state.messages.append({"role": "assistant", "content": answer_ia})
            with st.chat_message("assistant"):
                st.markdown(answer_ia)
                with st.expander("Sources found in the document"):
                    for font in result["source_documents"]:
                        st.info(f"trecho: ...{font.page_content[:250]}...")
else: 
    st.info("Please load a PDF document to start asking questions.")