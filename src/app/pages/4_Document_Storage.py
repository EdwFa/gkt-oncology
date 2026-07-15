import streamlit as st
import sys
import os
import json
from pathlib import Path
import pandas as pd

# Пути
root_path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.utils import inject_custom_css, setup_sidebar

st.set_page_config(page_title="Document Storage - GKT Oncology", page_icon="📁", layout="wide")
inject_custom_css()
setup_sidebar()

st.title("📁 Хранилище документов")
st.markdown("Просмотр исходных документов, статуса обработки и нарезанных чанков.")

raw_docs_dir = root_path / "data" / "raw_docs"
processed_docs_dir = root_path / "data" / "processed_docs"
chunks_path = root_path / "data" / "chunks" / "all_chunks.jsonl"

# Подсчитываем чанки
chunk_counts = {}
if chunks_path.exists():
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            # Извлекаем имя документа из chunk_id
            doc_name = c["chunk_id"].split("_chunk_")[0]
            chunk_counts[doc_name] = chunk_counts.get(doc_name, 0) + 1

if not raw_docs_dir.exists():
    st.warning("Директория с документами пуста или не существует.")
else:
    docs = []
    for file in raw_docs_dir.iterdir():
        if file.is_file() and file.suffix.lower() == '.pdf':
            doc_name = file.name
            
            # Проверяем обработку
            md_file = processed_docs_dir / f"{file.stem}_docling.md"
            status = "✅ Обработан" if md_file.exists() else "⏳ Ожидает"
            size_mb = file.stat().st_size / (1024 * 1024)
            num_chunks = chunk_counts.get(doc_name, 0)
            
            docs.append({
                "Документ": doc_name,
                "Размер (МБ)": f"{size_mb:.2f}",
                "Чанки": num_chunks,
                "Статус": status
            })
            
    if docs:
        df = pd.DataFrame(docs)
        
        st.markdown("**Выберите документ в таблице (кликните по строке), чтобы посмотреть его содержимое и чанки:**")
        
        # Интерактивная таблица
        event = st.dataframe(
            df, 
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        selected_rows = event.selection.rows
        if selected_rows:
            selected_idx = selected_rows[0]
            selected_doc = df.iloc[selected_idx]["Документ"]
            selected_stem = Path(selected_doc).stem
            
            st.divider()
            st.subheader(f"📄 Просмотр: {selected_doc}")
            
            tab1, tab2 = st.tabs(["🧩 Чанки", "📝 Полный Markdown"])
            
            with tab1:
                # Читаем чанки для этого документа
                doc_chunks = []
                if chunks_path.exists():
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            c = json.loads(line)
                            if c["chunk_id"].startswith(selected_doc):
                                doc_chunks.append(c)
                
                if doc_chunks:
                    st.info(f"Найдено чанков: **{len(doc_chunks)}**")
                    for i, c in enumerate(doc_chunks):
                        with st.expander(f"Чанк {i+1} | ID: {c.get('chunk_id')}"):
                            st.markdown(c.get("text", ""))
                            if "metadata" in c:
                                st.caption("Метаданные:")
                                st.json(c.get("metadata", {}))
                else:
                    st.warning("Для данного документа чанки пока не сгенерированы.")
                    
            with tab2:
                md_path = processed_docs_dir / f"{selected_stem}_docling.md"
                if md_path.exists():
                    try:
                        with open(md_path, 'r', encoding='utf-8') as f:
                            md_content = f.read()
                        st.markdown(md_content)
                    except Exception as e:
                        st.error(f"Ошибка при чтении файла: {e}")
                else:
                    st.warning("Markdown версия документа еще не создана.")
    else:
        st.info("PDF документы не найдены в хранилище (data/raw_docs).")
