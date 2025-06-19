# create_index.py
# Membangun FAISS index dari dataset mental health dengan Cohere + LangChain

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

def create_faiss_index():
    load_dotenv()

    # Path ke dataset CSV dan folder index
    csv_path = "data/metadata_who_data_for_idn.csv"        # Ganti dengan file kamu
    index_dir = "data/faiss_index"                         # Disimpan sebagai folder

    # Validasi file
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ File CSV tidak ditemukan: {csv_path}")

    # Ambil API key dari .env
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("❌ COHERE_API_KEY tidak ditemukan di file .env")

    # Set user agent (hindari error KeyError: user_agent)
    os.environ["LANGCHAIN_USER_AGENT"] = "mental-health-chatbot"

    # Baca data dan gabungkan ke dokumen
    df = pd.read_csv(csv_path).fillna("")
    df["combined"] = df.astype(str).agg(" ".join, axis=1)
    texts = df["combined"].tolist()
    docs = [Document(page_content=text) for text in texts]

    # Split dokumen jika panjang
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Buat embeddings
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-multilingual-v3.0",
        user_agent="mental-health-chatbot"
    )

    # Buat FAISS vectorstore dan simpan
    print("🔁 Membuat vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)

    print(f"✅ FAISS index berhasil disimpan ke folder: {index_dir}")

if __name__ == "__main__":
    create_faiss_index()
