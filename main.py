# File utama aplikasi Chatbot Kesehatan Mental AI.

import os
from dotenv import load_dotenv
import streamlit as st

# Import library LangChain dan lainnya
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import GoogleAPICallError
from googlesearch import search
import google.generativeai as genai
# Import komponen lokal
from retriever import FaissRetriever
from mental_health_processor import extract_mental_health_document
from callback_handler import GeminiCallbackHandler

# Import semua fungsi dari tools
from tools.date_tools import show_current_date
from tools.cooping_tools import get_coping_tips
from tools.pscyologist_tools import get_professional_help
from tools.save_history import save_chat_history
from tools.translate_tools import translate_to_indonesian, translate_to_english

# Load environment variables di awal
load_dotenv()

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
   
def show_typing_indicator():
    st.markdown("""
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """, unsafe_allow_html=True)


def get_Google_Search_results(query: str) -> str:
    """Melakukan pencarian Google dan mengembalikan string jawaban yang sudah diformat lengkap dengan URL."""
    try:
        num_results = 5
        print(f"Melakukan pencarian Google untuk '{query}'...")
        results = list(search(query, num_results=num_results, lang="id"))
        
        if not results:
            return f"Maaf, saya tidak dapat menemukan hasil yang relevan di Google untuk '{query}'."

        url_list = "\n".join([f"{i+1}. {url}" for i, url in enumerate(results)])
        final_answer = (
            f"Tentu, berikut adalah {len(results)} hasil pencarian teratas untuk '{query}':\n"
            f"{url_list}\n\n"
            "**Penting**: Harap evaluasi sendiri kredibilitas dan keakuratan informasi dari situs-situs tersebut."
        )
        return final_answer
        
    except Exception as e:
        return f"Terjadi kesalahan saat melakukan pencarian Google: {str(e)}"

# --- Logika Agent ---
def run_agent(user_input: str, retriever: FaissRetriever, memory, pdf_content: str = None):
    """Menginisialisasi dan menjalankan agent."""
    print("--- Menjalankan Unified Agent ---")
    
    system_prompt = ""
    if pdf_content:
        system_prompt = f"""
        PERHATIAN: Pengguna telah mengunggah sebuah dokumen kesehatan mental. Anda adalah asisten kesehatan mental yang sangat teliti dan komunikatif.
        
        TUGAS UTAMA ANDA:
        1. Jawab pertanyaan pengguna HANYA berdasarkan 'KONTEKS DOKUMEN' di bawah ini.
        2. Jika jawaban tidak ada di dokumen, katakan jujur bahwa informasi tersebut tidak ditemukan di dalam dokumen.
        3. Gunakan 'Tools' hanya jika pertanyaan jelas-jelas tidak berkaitan dengan isi dokumen.

        --- KONTEKS DOKUMEN ---
        {pdf_content}
        --- AKHIR KONTEKS DOKUMEN ---
        """

    final_input = f"{system_prompt}\n\nPertanyaan: {user_input}"

    try:
        tools = [
            Tool(name='cari_info_kesehatan_mental', func=lambda q: get_professional_help(q, retriever), description="Gunakan untuk menjawab pertanyaan spesifik tentang kesehatan mental dari database."),
            Tool(name='beri_rekomendasi_kesehatan_mental', func=lambda q: get_coping_tips(q, retriever), description="Gunakan untuk memberikan rekomendasi kesehatan mental berdasarkan topik dari database."),
            Tool(name='pencarian_internet_google', func=get_Google_Search_results, description="Gunakan HANYA untuk mencari informasi kesehatan mental yang SANGAT BARU."),
            Tool(name='terjemah_bahasa', func=translate_to_indonesian, description="Gunakan untuk menerjemahkan."),
            Tool(name='dapatkan_tanggal_sekarang', func=show_current_date, description="Gunakan untuk mengetahui tanggal dan waktu saat ini.")
        ]
        
        gemini_handler = GeminiCallbackHandler()
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.session_state.gemini_api_key,
            temperature=0)
        agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True, handle_parsing_errors="Maaf, terjadi sedikit kendala. Coba tanyakan dengan cara lain.")
        
        response = agent.run(final_input)
        return response

    except (GoogleAPICallError) as e:
        st.error(f"Error pada layanan Gemini: {e}. Pastikan API Key Anda valid.")
        raise e
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan agent: {e}")
        raise e

# --- Fungsi Utama Aplikasi ---
def main():
    st.set_page_config(page_title="Asisten Kesehatan Mental AI", page_icon="🧠", layout="centered")
    load_css()
    st.markdown("<header><h1>🧠 Asisten Kesehatan Mental Indonesia</h1><p>Didukung oleh AI dan Google Search</p></header>", unsafe_allow_html=True)
    
    initial_greeting = """Halo! Saya adalah Asisten Kesehatan Mental AI Anda.

**Apa yang bisa saya bantu?**

Anda bisa bertanya tentang data kesehatan mental umum dari database kami, seperti:
* `Apa statistik depresi di Indonesia?`
* `Bagaimana cara mengatasi kecemasan?`
* `Berikan rekomendasi untuk meningkatkan kesehatan mental.`

**Anda juga bisa mengunggah dokumen kesehatan mental (PDF) di sidebar.** Setelah diunggah, Anda bisa langsung bertanya tentang isinya, misalnya:
* `Apa rekomendasi dari dokumen ini?`
* `Sebutkan semua terapi yang dianjurkan dalam file tersebut.`
"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": initial_greeting}]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if "processed_file_name" not in st.session_state:
        st.session_state.processed_file_name = None
    if "pdf_content" not in st.session_state:
        st.session_state.pdf_content = None

    @st.cache_resource
    def init_retriever():
        retriever = FaissRetriever(index_path= "data/faiss_index")
    retriever = init_retriever()

    with st.sidebar:
        st.subheader("🔐 API Key Gemini")
        st.session_state.gemini_api_key = st.text_input("Masukkan API Key Gemini", type="password")
        if not st.session_state.gemini_api_key:
            st.error("Masukkan API Key Gemini dulu.")
            return
        st.header("Analisis Dokumen")
        uploaded_file = st.file_uploader("Upload PDF Dokumen Kesehatan Mental", type=['pdf'], key="pdf_uploader")
        if uploaded_file and uploaded_file.name != st.session_state.get('processed_file_name'):
            with st.spinner("Memproses dokumen..."):
                doc_info = extract_mental_health_document(uploaded_file)
                if 'error' in doc_info:
                    st.error(doc_info['error'])
                    st.session_state.processed_file_name = None
                else:
                    st.session_state.pdf_content = doc_info.get('full_text')
                    st.session_state.processed_file_name = uploaded_file.name
                    st.success("Dokumen berhasil diproses!")
                    st.session_state.messages.append({"role": "system", "content": f"Dokumen '{uploaded_file.name}' telah diunggah. Anda sekarang bisa bertanya mengenai isinya."})
        
        st.divider()
        with st.expander("📜 Riwayat Percakapan"):
            if not st.session_state.messages: st.write("Belum ada percakapan.")
            else:
                for msg in st.session_state.messages:
                    if msg["role"] != "system": st.markdown(f'**{msg["role"].replace("user", "Anda").replace("assistant", "AI")}:** *{msg["content"][:40]}...*')
        
        if st.button("Hapus Riwayat & Dokumen", type="secondary", key="delete_history"):
            st.session_state.messages = [{"role": "assistant", "content": "Riwayat chat dan dokumen telah dihapus. Silakan mulai percakapan baru."}]
            st.session_state.memory.clear()
            st.session_state.pdf_content = None
            st.session_state.processed_file_name = None
            st.rerun()

    for message in st.session_state.messages:
        if message.get("role") != "system":
            avatar = "🧑‍💻" if message["role"] == "user" else "🧠"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    if user_input := st.chat_input("Tanyakan sesuatu..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Asisten sedang berpikir..."):
                try:
                    pdf_context = st.session_state.get("pdf_content")
                    response_text = run_agent(user_input, retriever, st.session_state.memory, pdf_content=pdf_context)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Maaf, terjadi kesalahan fatal: {e}")
                    st.session_state.messages.pop()

if __name__ == "__main__":
    main()
