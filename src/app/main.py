import streamlit as st
import json
from pathlib import Path
import os
import sys

# Добавляем корень проекта в sys.path
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.app.utils import inject_custom_css, render_metric_card, setup_sidebar

st.set_page_config(
    page_title="GKT Oncology - QLoRA Dashboard",
    page_icon="🧬",
    layout="wide"
)

inject_custom_css()
setup_sidebar()

st.markdown("""
<div class="title-container">
    <h1>🧬 GKT Oncology QLoRA</h1>
    <h3>Панель управления пайплайном генерации датасета</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Здесь вы можете просматривать промежуточные и финальные результаты работы автономных агентов.
""")

# Сбор статистики
data_dir = Path("data")
raw_docs_dir = data_dir / "raw_docs"
processed_docs_dir = data_dir / "processed_docs"
chunks_file = data_dir / "chunks" / "all_chunks.jsonl"
ner_file = processed_docs_dir / "ner_pilot_results.json"
qa_dataset_file = data_dir / "qa_dataset" / "raw_dataset.jsonl"

col1, col2, col3, col4 = st.columns(4)

with col1:
    raw_docs_count = len(list(raw_docs_dir.glob("*.pdf"))) if raw_docs_dir.exists() else 0
    render_metric_card("Сырые документы", str(raw_docs_count), "📄")

with col2:
    chunks_count = sum(1 for _ in open(chunks_file)) if chunks_file.exists() else 0
    render_metric_card("Всего чанков", str(chunks_count), "✂️")

with col3:
    ner_count = 0
    if ner_file.exists():
        try:
            ner_data = json.load(open(ner_file))
            ner_count = len(ner_data)
        except:
            pass
    render_metric_card("Чанки с NER", str(ner_count), "🧠")

with col4:
    qa_count = sum(1 for _ in open(qa_dataset_file)) if qa_dataset_file.exists() else 0
    render_metric_card("QA Пары", str(qa_count), "💬")

st.divider()

st.subheader("Статус фоновых процессов")
st.info("Фоновые процессы (нарезка, NER, генерация QA) запускаются через терминал или оркестратор. Этот UI предназначен для визуализации результатов.", icon="ℹ️")

st.markdown("""
### Навигация
Используйте левое меню для навигации по этапам пайплайна:
* **Data Pipeline**: Просмотр текстов чанков и извлеченной медицинской онтологии (препараты, диагнозы).
* **QA Viewer**: Просмотр сырых и верифицированных QA пар, а также оценок LLM-ансамбля.
* **Sandbox**: Ручное тестирование агентов (вставьте текст и получите сгенерированные вопросы).
""")
