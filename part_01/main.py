from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
import logging

load_dotenv()

path_db = "db"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

prompt_template_explicação = """ Responda essa pergunta com base nessas informações """
prompt_template = """ 
Responda a está pergunta: 
{question} 

com base nessas informações:
{knowledge_base}
"""

def ask():
    question = input("Digite sua pergunta: ")
    embedding_function = HuggingFaceEndpointEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=path_db, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(question, k=4)
    if len(results) == 0 or results[0][1] < 0.2:
        logger.info("Not enough relevant information found in the database.")
        return
    
    results_text = []
    for result in results:
        text = result[0].page_content
        results_text.append(text)

    knowledge_base = "\n\n------\n\n".join(results_text)

    prompt = PromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"question": question, "knowledge_base": knowledge_base})

    llm_endpoint = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.3)
    model = ChatHuggingFace(llm = llm_endpoint)
    text_response =  model.invoke(prompt).content
    print("Answer: ", text_response)

ask()
