"""
Страница просмотра сгенерированного QA-датасета и оценок верификатора.
Выводит удобную таблицу со статусами верификации, вопросами, ответами и 
подробными оценками от каждой LLM в ансамбле.
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import sys
root_path: Path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.utils import inject_custom_css, setup_sidebar

st.set_page_config(page_title="QA Viewer - GKT Oncology", page_icon="🤖", layout="wide")
inject_custom_css()
setup_sidebar()

st.markdown("""
<div class="title-container">
    <h1>🤖 QA Dataset Viewer</h1>
    <h3>Просмотр результатов работы Мультиагентной системы и оценок ансамбля</h3>
</div>
""", unsafe_allow_html=True)

qa_file: Path = Path("data/qa_dataset/raw_dataset.jsonl")

@st.cache_data
def load_qa_data() -> List[Dict[str, Any]]:
    """
    Загружает QA-датасет из файла raw_dataset.jsonl.
    
    Returns:
        List[Dict[str, Any]]: Список словарей, каждый из которых представляет QA-пару.
    """
    qa_pairs: List[Dict[str, Any]] = []
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    qa_pairs.append(json.loads(line))
                except Exception:
                    continue
    return qa_pairs

qa_pairs = load_qa_data()

if not qa_pairs:
    st.warning("QA-датасет пока пуст. Запустите оркестратор.")
else:
    # Конвертация в pandas DataFrame для удобного просмотра и фильтрации
    df_data: List[Dict[str, Any]] = []
    for qa in qa_pairs:
        v: Dict[str, Any] = qa.get("verification", {})
        passed: bool = v.get("passed", False)
        
        # Определение статуса: отказ A-Agent'а, успешно пройдено или провалено ансамблем
        if qa.get("refusal", False):
            status = "Refusal"
        elif passed:
            status = "Passed ✅"
        else:
            status = "Failed ❌"
            
        df_data.append({
            "chunk_id": qa.get("chunk_id"),
            "question_type": qa.get("question_type"),
            "status": status,
            "s_final": round(v.get("s_final", 0.0), 2) if "s_final" in v else 0.0,
            "confidence": round(v.get("confidence", 0.0), 2) if "confidence" in v else 0.0,
            "consensus": round(v.get("consensus", 0.0), 2) if "consensus" in v else 0.0,
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "model_scores": v.get("model_scores", {})
        })
        
    df = pd.DataFrame(df_data)
    
    st.sidebar.header("Фильтры")
    selected_status: List[str] = st.sidebar.multiselect("Статус", df["status"].unique(), default=df["status"].unique())
    selected_types: List[str] = st.sidebar.multiselect("Тип вопроса", df["question_type"].unique(), default=df["question_type"].unique())
    
    filtered_df = df[df["status"].isin(selected_status) & df["question_type"].isin(selected_types)]
    
    st.metric("Отображено пар", len(filtered_df))
    
    # Отрисовка сетки (карточек) для каждой QA пары
    for i, row in filtered_df.iterrows():
        with st.container():
            col_status, col_q, col_a, col_metrics = st.columns([1, 3, 3, 2])
            
            with col_status:
                if "Passed" in row["status"]:
                    st.success(row["status"])
                elif "Failed" in row["status"]:
                    st.error(row["status"])
                else:
                    st.warning(row["status"])
                st.caption(f"Тип: {row['question_type']}")
                st.caption(f"Чанк: {row['chunk_id']}")
                
            with col_q:
                st.markdown("**Вопрос**")
                st.info(row["question"])
                
            with col_a:
                st.markdown("**Ответ (A-Agent)**")
                st.success(row["answer"])
                
            with col_metrics:
                st.markdown("**Метрики Ансамбля**")
                st.write(f"**S_final:** {row['s_final']}")
                st.write(f"**Confidence:** {row['confidence']}")
                st.write(f"**Consensus:** {row['consensus']}")
                with st.expander("Оценки моделей"):
                    st.json(row["model_scores"])
            
        st.divider()
