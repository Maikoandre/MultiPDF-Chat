from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import tempfile, os
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


promptTemplate = '''
Answer the following question as best as you can using the provided context.

User question: {question}
Context: {context}
If you don't know the answer, just say that you don't know, don't try to make up an answer.
'''

def load_model_embedding():
    return HuggingFaceEndpointEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )

def load_model_chat():
    llm_endpoint = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.3)

    return ChatHuggingFace(llm = llm_endpoint)

def process_pdf(file_pdf, embedding_model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_pdf.getvalue())
            temp_pdf_path = temp_pdf.name
        
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(
            chunks, 
            embedding_model
        )

        os.remove(temp_pdf_path)
        return vector_db
    except Exception as e:
        logger.error(f"Deu ruim kkkkk: {e}")
        return None
    
def create_conversation_chain(vector_db, llm_model, chat_memory):
    retriever = vector_db.as_retriever(search_kwargs={"k":5})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=retriever,
        memory=chat_memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': PromptTemplate.from_template(promptTemplate)}
    )
    return conversation_chain