from langchain_community.llms import HuggingFaceEndpoint

from dotenv import load_dotenv

import os

load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore

# Retrieve Pinecone API key from environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pincone embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Define Pinecone index name
index_name = "index"

vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)


path = "try2\Modi-Ki-Guarantee-Sankalp-Patra-English_2.pdf"

def data_stuff(path):

    # Load PDF document
    loader = PyPDFLoader(path)
    data = loader.load()
    # print(data)

    # Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(data)
    # len(text_chunks)

    # Initialize Pinecone vector store
    vector_store_from_docs = PineconeVectorStore.from_documents(text_chunks, index_name=index_name, embedding=embeddings)


query = "How has the Indian government empowered farmers through various initiatives, including MSP hikes, procurement, and income support programs?"

def ask_query(query):

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_store.as_retriever())
    result = qa.invoke(query)
    return result



# ask_query(query)


# flask code here

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


app = Flask(__name__)

CORS(app , supports_credentials=True)

@app.route('/ask', methods=['GET'])

@cross_origin(supports_credentials=True)

def query_user():
    question = request.args.get('query')
    answer = ask_query(question)
    return jsonify({"answer": answer})


@app.route('/upload', methods=['GET', 'POST'])

@cross_origin(supports_credentials=True)

def upload_document():
    if request.method == 'POST':

        if 'file' not in request.files:
            print('dost... kya hai ye?')
            return jsonify({"status": 'failed'})
        

        file = request.files['file']
        file_name = file.filename
        filepath = f"backend/try2/uploads/{file_name}"
        file.save(filepath)
        try:
            data_stuff(filepath)
            return jsonify({"status":'yay'})
        except Exception as e:
            return jsonify({"status":'failed'})


if __name__=='__main__':
    app.run(debug=True)


