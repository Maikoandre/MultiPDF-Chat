# Multi-PDF Chat

This project allows you to chat with your PDF documents. It's a great way to quickly find information and get answers from your files.

---

## How it Works

The application uses a combination of technologies to make this possible:

* **Streamlit:** For creating the user-friendly web interface.
* **LangChain:** A framework for developing applications powered by language models.
* **Hugging Face:** For providing the language models (for both embeddings and chat).
* **Chroma:** As the vector store to save the document embeddings.

The process is as follows:

1.  **Upload a PDF:** You start by uploading a PDF file through the web interface.
2.  **Document Processing:** The application processes the PDF, splitting it into smaller chunks of text.
3.  **Embeddings:** Each chunk of text is then converted into a numerical representation called an "embedding" using a Hugging Face model.
4.  **Vector Store:** These embeddings are stored in a Chroma vector database. This allows for efficient searching of the most relevant text chunks based on your questions.
5.  **Chat:** When you ask a question, the application searches the vector store for the most relevant text chunks and uses them as context for a large language model from Hugging Face to generate an answer.

---

## Project Structure

The project is divided into two main parts:

### Part 1: Database Creation (`create_database.py`)

This script is used to process a directory of PDF files and create a persistent vector store.

* `load_documents()`: Loads all PDF files from a specified directory.
* `split_documents()`: Splits the loaded documents into smaller chunks.
* `vectorize_chunks()`: Creates embeddings for the chunks and saves them in a Chroma database.

### Part 2: The Web Application (`app.py` and `engine.py`)

This is the main Streamlit application that you interact with.

* **`app.py`:** This file contains the code for the Streamlit user interface. It handles file uploads, displays the chat interface, and manages the conversation state.

* **`engine.py`:** This file contains the core logic for processing the PDF and handling the conversation.
    * `load_model_embedding()`: Loads the sentence-transformer model for creating embeddings.
    * `load_model_chat()`: Loads the language model for generating answers.
    * `process_pdf()`: Takes an uploaded PDF file, processes it, and creates an in-memory vector store.
    * `create_conversation_chain()`: Creates the LangChain conversation chain that connects the language model, retriever, and chat memory.

---

## How to Run

### Prerequisites

You need to have Python installed on your system. You will also need to install the required libraries. You can do this by running:

```bash
pip install streamlit python-dotenv langchain langchain_community langchain-chroma langchain-huggingface pypdf
