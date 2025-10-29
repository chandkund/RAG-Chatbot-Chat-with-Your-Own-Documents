# Vectore_Store.py
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import WebBaseLoader
import faiss
import bs4

# Constants
model = "granite3-moe:1b"
llm = ChatOllama(model=model)
embeddings_model = OllamaEmbeddings(model=model)
VECTOR_STORE_PATH = Path("faiss_index")


class VectorStore:
   
    def __init__(self, vector_store_path=VECTOR_STORE_PATH, llm_model="granite3-moe:1b",
                  chunk_size=500, chunk_overlap=50, persist=True, index_path="faiss_index"):
      
        self.vector_store_path = vector_store_path
        self.llm_model = llm_model
        self.embeddings_model = OllamaEmbeddings(model=llm_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist = persist
        self.index_path = index_path

        self._setup_vector_store()
    

    def _setup_vector_store(self) -> None:
       
        print(self.vector_store_path)
        if self.vector_store_path.exists():
            self.vector_store =  FAISS.load_local(str(self.vector_store_path), 
                                    embeddings=self.embeddings_model,
                                     allow_dangerous_deserialization=True)
        else:
            print(f"No Vectorstore found at {str(self.vector_store_path)}")
            try:
                self.embedding_dim = len(self.embeddings_model.embed_query("hello world"))
            except Exception as e:
                logging.error(f"Embedding model '{self.llm_model}' not found or not loaded properly: {e}")
                self.embedding_dim = 1536

            self.index = faiss.IndexFlatL2(self.embedding_dim)

            self.vector_store = FAISS(
                embedding_function=self.embeddings_model,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

            if self.persist:
                self.vector_store.save_local(self.index_path)
    

    def load_documents(self, data_path) -> List[Document]:
       
        documents = []
        for pdf_path in Path(data_path).glob("*.pdf"):
            docs = self.load_document(pdf_path)
            print(len(docs))
            documents.extend(docs)
        return documents
    
    def add_documents(self, documents: List[Document]) -> List[Document]:
        
        splitted_docs = self.chunk_documents(documents=documents)
        self.vector_store.add_documents(splitted_docs or [])
        self.vector_store.save_local(self.index_path)
        return splitted_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                              chunk_overlap=self.chunk_overlap)    
        return text_splitter.split_documents(documents)
    
    
    def load_document(self, pdf_path: Path) -> List[Document]:
        
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        return docs if isinstance(docs, list) else [docs]
    
    def add_document(self, filePath: Path) -> List[Document]:
       
        docs = self.load_document(filePath)
        return self.add_documents(docs)
    
    def similarity_search(self, question: str, k:int) -> List[Document]:
        
        return self.vector_store.similarity_search(question, k=k)
    
    def as_retriever(self) -> VectorStoreRetriever:
       
        return self.vector_store.as_retriever()

    def index_websites(self, urls: list[str]) -> List[Document]:
       
        docs = self.website_to_documents(urls)
        return self.add_documents(docs)

    def website_to_documents(self, urls: list[str]) -> list[Document]:
        
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        return docs