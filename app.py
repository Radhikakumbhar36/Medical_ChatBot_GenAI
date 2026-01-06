from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

app = Flask(__name__)
load_dotenv()

# ENV
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medicalbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")
        response = rag_chain.invoke({"input": msg})
        return response["answer"]
    except Exception as e:
        print("ERROR:", e)
        return "Error occurred."

if __name__ == "__main__":
    app.run(debug=True)
