# mental_health_chatbot/utils/rag_helper.py
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

def load_retriever():
    embeddings = GooglePalmEmbeddings()
    db = FAISS.load_local("data/faiss_index", embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

def get_rag_response(query, retriever, llm):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa({"query": query})
