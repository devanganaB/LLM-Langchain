from langchain_community.llms import HuggingFaceEndpoint

# from getpass import getpass
from dotenv import load_dotenv

# HUGGINGFACEHUB_API_TOKEN = getpass()

import os

load_dotenv()


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


from langchain_pinecone import PineconeVectorStore

# from langchain_text_splitters import CharacterTextSplitter



# Retrieve Pinecone API key from environment variable
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")



# Load PDF document
loader = PyPDFLoader("try2\Modi-Ki-Guarantee-Sankalp-Patra-English_2.pdf")
data = loader.load()
# print(data)

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
# len(text_chunks)

# Initialize Pincone embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# since pinecone is being used for vector store
# vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)


# Define Pinecone index name
index_name = "myindex"

# Initialize Pinecone vector store
vector_store = PineconeVectorStore.from_documents(text_chunks, index_name=index_name, embedding=embeddings)

# Add documents to Pinecone index
# vector_store.add_documents(text_chunks)

#  not in use so far
# ---------------------------------------------------------------
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

# ----------------------------------------------------------------

query = "How has the Indian government empowered farmers through various initiatives, including MSP hikes, procurement, and income support programs?"

# result_similar = vector_store.similarity_search(query)
# print('SEARCH KA RESULTTTTTTTTTTTTTTT')
# print(result_similar)



# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_store.as_retriever(search_kwargs={"k": 2}))

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_store.as_retriever()  )
result = qa.invoke(query)
print(result)

