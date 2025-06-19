# retriever.py

import os
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS

  
class FaissRetriever:
    def __init__(self, index_path: str):
        cohere_api_key = os.getenv("COHERE_API_KEY")  # pastikan ini ada di .env atau diset secara manual
        self.embedding_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-multilingual-v3.0",
            user_agent="mental-health-chatbot"
        )
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index FAISS tidak ditemukan di: {index_path}")
        
        self.vectorstore = FAISS.load_local(
              index_path,
              self.embedding_model,
              allow_dangerous_deserialization=True  # Hati-hati dengan keamanan!
          )
  