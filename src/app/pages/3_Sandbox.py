"""
Интерактивная песочница для тестирования агентов Q-Agent и A-Agent вручную.
Пользователь может вставить любой текст и симулировать пайплайн без сохранения в БД.
"""
import streamlit as st
import asyncio
import aiohttp
import sys
from pathlib import Path
from typing import Dict, Any

# Добавляем корень проекта в sys.path, чтобы работал импорт из src
root_path: Path = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_path))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app.utils import inject_custom_css, setup_sidebar
from src.qa_generation.q_agent import QuestionGenerator
from src.qa_generation.a_agent import AnswerGenerator

st.set_page_config(page_title="Sandbox - GKT Oncology", page_icon="🧪", layout="wide")
inject_custom_css()
setup_sidebar()

st.markdown("""
<div class="title-container">
    <h1>🧪 Интерактивная Песочница</h1>
    <h3>Ручное тестирование Агентов (Генератора Вопросов и Ответов) на произвольном тексте</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Входные данные")
    input_text: str = st.text_area(
        "Текст клинической рекомендации (или фрагмент):", 
        height=300, 
        placeholder="Вставьте текст сюда..."
    )
    input_entities: str = st.text_area(
        "Медицинские сущности (JSON):", 
        height=150, 
        value='{"препараты": ["Леватиниб"], "диагнозы": ["Рак щитовидной железы"]}'
    )
    
    run_btn = st.button("Запустить Q-A Пайплайн 🚀", type="primary", use_container_width=True)

with col2:
    st.subheader("Результат")
    res_container = st.container()

async def run_pipeline(text: str, entities_str: str) -> str | None:
    """
    Асинхронная симуляция пайплайна генерации для песочницы.
    
    Args:
        text (str): Текст чанка для тестирования.
        entities_str (str): Строка в формате JSON с сущностями.
        
    Returns:
        Optional[str]: Строка с ошибкой, если она произошла. Иначе None.
    """
    try:
        entities: Dict[str, Any] = eval(entities_str) if entities_str else {}
    except Exception as e:
        return f"Ошибка парсинга сущностей: {e}"
        
    chunk: Dict[str, Any] = {
        "text": text,
        "entities": entities
    }
    
    q_agent = QuestionGenerator()
    a_agent = AnswerGenerator()
    
    # 1. Генерация вопросов
    with st.spinner("Q-Agent генерирует вопросы..."):
        async with aiohttp.ClientSession() as session:
            q_batch = await q_agent.generate_questions(session, chunk)
            
    if not q_batch or not q_batch.questions:
        return "Не удалось сгенерировать вопросы. Проверьте логи."
        
    res_container.success(f"Q-Agent сгенерировал {len(q_batch.questions)} вопросов!")
    
    # 2. Генерация ответов
    with st.spinner("A-Agent отвечает..."):
        async with aiohttp.ClientSession() as session:
            tasks = [a_agent.generate_answer(session, text, q) for q in q_batch.questions]
            a_responses = await asyncio.gather(*tasks)
            
    # Вывод результатов в UI
    for i, (q, a) in enumerate(zip(q_batch.questions, a_responses)):
        with res_container.expander(f"Q{i+1}: [{q.type}] {q.text}"):
            if a.refusal:
                st.error(a.text)
            else:
                st.success(a.text)
                
    return None

if run_btn:
    if not input_text:
        st.warning("Пожалуйста, вставьте текст!")
    else:
        error_msg = asyncio.run(run_pipeline(input_text, input_entities))
        if error_msg:
            st.error(error_msg)
