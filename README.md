# Document Retrieval and Question Answering Application
This project is a document retrieval and question-answering system built using Langchain, HuggingFace, Pinecone, and Streamlit. The application allows users to upload PDF documents, processes them to create embeddings, and then retrieves relevant information from the documents based on user queries. The document's content is indexed in Pinecone, and HuggingFace's language models are used to answer questions by retrieving the most relevant text chunks.

## Features
### PDF Upload: Upload a PDF document that is processed and indexed for retrieval.
### Document Chunking: The document is split into smaller chunks using recursive text splitting for efficient retrieval.
### Embedding and Vector Store: HuggingFace embeddings are used to create vectors for the document's text, stored in a Pinecone index.
### Question Answering: A question is answered by retrieving relevant document sections from the Pinecone vector store.
### Streamlit Interface: The application provides a user-friendly interface built with Streamlit.

## Project Structure
### HuggingFaceEmbeddings: Used to create embeddings from text chunks for storing in Pinecone.
### PyPDFLoader: Loads and processes PDF documents.
### PineconeVectorStore: Handles the vector store functionality using Pinecone.
### Streamlit: Provides the interface for file uploading and question answering.
### RetrievalQA: Performs the retrieval and question-answering process using HuggingFace's language model.

## Setup Instructions
### Prerequisites
* Python 3.8+
* A Pinecone account for API access and an API key.
* A HuggingFace account for API access to language models.
* A .env file with the following environment variables:
* HUGGINGFACEHUB_API_TOKEN: Your HuggingFace API key.
* PINECONE_API_KEY: Your Pinecone API key.

### Installation
1. Clone the repository.
2. Install the required Python packages:

`pip install -r requirements.txt`

3. Create a .env file in the root directory with your API tokens:

`HUGGINGFACEHUB_API_TOKEN=<your_huggingface_api_token>`
`PINECONE_API_KEY=<your_pinecone_api_key>`

4. Run the application:

`streamlit run app.py`

### Usage
* Upload a PDF document using the "Upload a PDF document" section.
* Once the document is uploaded and processed, type your query in the text input box.
* Click the "Ask" button, and the application will display the retrieved and generated response.


### Key Functions
* `load_document(path)`: Loads and splits the document into chunks, stores them as embeddings in Pinecone.
* `ask_query(query)`: Takes a user query and retrieves relevant information from the document using HuggingFace's language model.


### Future Improvements
- Add support for multiple file formats (e.g., DOCX).
- Implement additional LLM models for more diverse querying options.
- Enhance the front-end interface for better user experience.


### License
- This project is open-source and available under the MIT License.

- This README serves as a guide to understanding the purpose, setup, and usage of the Document Retrieval and Question Answering Application.









![image](https://github.com/user-attachments/assets/e3e523d6-1fd6-4b4a-ab26-73c18966df6f)
