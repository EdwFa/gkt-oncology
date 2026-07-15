"""
Страница Evaluation.
Показывает результаты работы LLM-Судьи по оценке LoRA-адаптера.
Позволяет интерактивно пообщаться с обученной локальной моделью (если запущен MLX).
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
import sys

# Настройка путей
root_path: Path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.utils import inject_custom_css, setup_sidebar, render_metric_card

# Попытка импортировать MLX для локального чата
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

st.set_page_config(page_title="Evaluation - GKT Oncology", page_icon="⚖️", layout="wide")
inject_custom_css()
setup_sidebar()

st.markdown("""
<div class="title-container">
    <h1>⚖️ Оценка и Тестирование (Evaluation)</h1>
    <h3>Отчет LLM-Судьи (Qwen2.5) и интерактивный чат с обученной моделью</h3>
</div>
""", unsafe_allow_html=True)

report_file = Path("data/qa_dataset/evaluation_report.json")

tab1, tab2 = st.tabs(["📊 Отчет Судьи", "💬 Чат с LoRA-моделью"])

with tab1:
    st.subheader("Результаты прогона валидационной выборки")
    
    if not report_file.exists():
        st.info("Отчет об оценке пока не сгенерирован. Запустите `src/evaluation/evaluate.py`.")
    else:
        with open(report_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        avg_score = report_data.get("average_score", 0.0)
        total_eval = report_data.get("total_evaluated", 0)
        details = report_data.get("details", [])
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            render_metric_card("Средний балл (из 5)", f"{avg_score:.2f}", "⭐")
        with col_m2:
            render_metric_card("Оценено вопросов", str(total_eval), "📝")
            
        st.divider()
        st.markdown("### Подробные результаты по каждому вопросу")
        
        # Конвертируем в DataFrame для красивой таблицы
        if details:
            df = pd.DataFrame(details)
            
            # Добавим цветные иконки для баллов
            def get_score_icon(score):
                if score == 5: return "⭐⭐⭐⭐⭐"
                if score == 4: return "⭐⭐⭐⭐"
                if score == 3: return "⭐⭐⭐"
                if score == 2: return "⭐⭐"
                return "⭐"
                
            df["Рейтинг"] = df["score"].apply(get_score_icon)
            
            for i, row in df.iterrows():
                with st.expander(f"[{row['score']}/5] {row['question'][:100]}..."):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Эталонный ответ:**")
                        st.info(row["reference"])
                    with c2:
                        st.markdown("**Ответ нашей LoRA модели:**")
                        st.success(row["prediction"])
                    
                    st.markdown("**Вердикт судьи (Qwen2.5-Turbo):**")
                    st.warning(row["reasoning"])

with tab2:
    st.subheader("Интерактивное тестирование (Playground)")
    if not MLX_AVAILABLE:
        st.error("MLX Framework недоступен. Запустите приложение на Mac Apple Silicon с установленным `mlx-lm`.")
    else:
        st.info("Внимание: Модель загружается в оперативную память Mac (Unified Memory).")
        
        if "model_loaded" not in st.session_state:
            st.session_state.model_loaded = False
            
        if not st.session_state.model_loaded:
            if st.button("Загрузить LoRA-модель в память", type="primary"):
                with st.spinner("Загрузка весов (может занять 10-20 секунд)..."):
                    try:
                        st.session_state.model, st.session_state.tokenizer = load("Qwen/Qwen2.5-7B-Instruct", adapter_path="adapters/")
                        st.session_state.model_loaded = True
                        st.success("Модель успешно загружена!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка загрузки: {e}")
        else:
            st.success("Модель активна и готова к работе.")
            
            # Чат интерфейс
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Задайте вопрос по онкологии ЖКТ..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Генерация ответа..."):
                        # Форматируем промпт
                        formatted_prompt = st.session_state.tokenizer.apply_chat_template(
                            st.session_state.messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        # Генерация
                        response = generate(
                            st.session_state.model, 
                            st.session_state.tokenizer, 
                            prompt=formatted_prompt, 
                            max_tokens=512, 
                            verbose=False
                        )
                        st.markdown(response)
                        
                st.session_state.messages.append({"role": "assistant", "content": response})
