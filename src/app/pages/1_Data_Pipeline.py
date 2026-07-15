import streamlit as st
import json
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.utils import inject_custom_css, setup_sidebar

st.set_page_config(page_title="Data Pipeline - GKT Oncology", page_icon="📑", layout="wide")
inject_custom_css()
setup_sidebar()

st.markdown("""
<div class="title-container">
    <h1>📑 Data Pipeline Viewer</h1>
    <h3>Просмотр результатов нарезки документов (Chunking) и извлечения сущностей (NER)</h3>
</div>
""", unsafe_allow_html=True)

chunks_file = Path("data/chunks/all_chunks.jsonl")
ner_file = Path("data/processed_docs/ner_pilot_results.json")

@st.cache_data
def load_data():
    chunks = {}
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                c = json.loads(line)
                chunks[c["chunk_id"]] = c
                
    ner_data = {}
    if ner_file.exists():
        try:
            with open(ner_file, 'r', encoding='utf-8') as f:
                ner_results = json.load(f)
                for res in ner_results:
                    ner_data[res["chunk_id"]] = res.get("entities", {})
        except Exception as e:
            st.error(f"Error loading NER data: {e}")
            
    return chunks, ner_data

chunks, ner_data = load_data()

if not chunks:
    st.warning("Чанки не найдены. Сначала запустите пайплайн обработки.")
else:
    chunk_ids = list(chunks.keys())
    
    st.sidebar.header("Фильтры")
    selected_chunk_id = st.sidebar.selectbox("Выберите чанк для просмотра:", chunk_ids)
    
    chunk = chunks[selected_chunk_id]
    entities = ner_data.get(selected_chunk_id, {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Чанк: {selected_chunk_id}")
        st.caption(f"Источник: {chunk.get('source_doc', 'Unknown')}")
        st.text_area("Текст чанка", value=chunk.get("text", ""), height=400, disabled=True)
        
    with col2:
        st.subheader("Извлеченная Онтология (NER)")
        if not entities:
            st.info("NER не производился для данного чанка.")
        else:
            if "error" in entities:
                st.error("Ошибка извлечения NER")
                st.json(entities)
            else:
                for category, items in entities.items():
                    with st.expander(f"{category.capitalize()} ({len(items) if isinstance(items, list) else 1})"):
                        if isinstance(items, list):
                            for item in items:
                                st.markdown(f"- {item}")
                        else:
                            st.write(items)
