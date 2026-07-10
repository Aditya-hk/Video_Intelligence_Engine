from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_core.documents import Document

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=Chroma(
    collection_name="video_rag",
    embedding_function=embedding_model,
    persist_directory=Path("data/vectorstore")
)

def add_to_vectorstore(chunks):
    documents=[]

    for chunk in chunks:
        documents.append(Document(
            page_content=chunk["text"],
            metadata={
                "start": chunk["start"],
                "end": chunk["end"]
            }
        ))
    vector_store.add_documents(documents)

def retrieve_from_vectorstore(query, k=3):
    
    results=vector_store.similarity_search(query, k=k)
    return results

    
