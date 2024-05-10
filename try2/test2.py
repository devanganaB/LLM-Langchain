from langchain_community.llms import HuggingFaceEndpoint

# from getpass import getpass
from dotenv import load_dotenv

# HUGGINGFACEHUB_API_TOKEN = getpass()

import os

load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# question = "Who won the FIFA World Cup in the year 1994? "

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate.from_template(template)



repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=500, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
)
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# print(llm_chain.run(question))



from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load PDF document
loader = PyPDFLoader("try2\KakushIN_problem_statement.pdf")
data = loader.load()
print(data)

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
len(text_chunks)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

query = "what is the man to women ratio?"

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))
result = qa.invoke(query)
print(result)

