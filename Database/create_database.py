from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import logging

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

path_db = "db"

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    logger.info(f"First chunk preview: {chunks[0].page_content[:500]}...")
    vectorize_chunks(chunks)

def load_documents():
    loader = PyPDFDirectoryLoader(path_db, glob="*.pdf")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from {path_db}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks

def vectorize_chunks(chunks):
    embeddings = HuggingFaceEndpointEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2"
    )
    Chroma.from_documents(
        chunks, 
        embeddings,
        persist_directory="db"
    )
    logger.info("Vector store created and persisted.")


if __name__ == "__main__":
    main()