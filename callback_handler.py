# mental_health_chatbot/callback_handler.py
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from typing import Any, Dict, List, Optional
import time

class GeminiCallbackHandler(BaseCallbackHandler):
    """Custom callback handler untuk Gemini dengan visualisasi Streamlit"""
    
    def __init__(self):
        self.container = st.empty()
        self.response = ""
        self.last_update = time.time()
        
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Menampilkan loading state saat LLM mulai memproses"""
        with self.container:
            st.markdown("🔄 <span class='thinking'>Memikirkan respons terbaik...</span>", unsafe_allow_html=True)
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Streaming token per token dengan typing effect"""
        self.response += token
        
        # Limit update rate untuk performance
        if time.time() - self.last_update > 0.1:
            self._update_display()
            self.last_update = time.time()
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Final update saat semua token selesai"""
        self._update_display(final=True)
        
    def _update_display(self, final=False):
        """Update tampilan Streamlit"""
        with self.container:
            if final:
                st.markdown(f"""
                <div class='assistant-message'>
                    <div class='message-header'>🧠 Asisten Kesehatan Mental</div>
                    <div class='message-content'>{self.response}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='assistant-message'>
                    <div class='message-header'>🧠 Asisten Kesehatan Mental</div>
                    <div class='message-content'>{self.response}▌</div>
                </div>
                """, unsafe_allow_html=True)
    
    def on_error(self, error: BaseException, **kwargs: Any) -> None:
        """Menangani error dengan tampilan yang baik"""
        with self.container:
            st.error(f"⚠️ Gangguan pada sistem: {str(error)}")
