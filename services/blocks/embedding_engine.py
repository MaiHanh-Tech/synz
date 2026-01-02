from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_encoder(model_name="paraphrase-MiniLM-L12-v2", device='cpu'):
    try:
        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = 128
        return model
    except Exception:
        return None

def encode_texts(encoder, texts):
    if encoder is None: return []
    return encoder.encode(texts)
