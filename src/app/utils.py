"""
Вспомогательный модуль (утилиты) для Streamlit-приложения.
Содержит функции для инъекции кастомного CSS, настройки сайдбара и рендера UI-компонентов.
"""
import streamlit as st

def inject_custom_css() -> None:
    """Внедряет пользовательские CSS-стили в приложение для улучшения внешнего вида."""
    st.markdown("""
    <style>
        .main {
            padding: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .title-container {
            border-radius: 10px;
            padding: 20px;
            background: linear-gradient(90deg, #1E3B70 0%, #29539B 100%);
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .card {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .agent-icon {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: bold;
            color: white;
            background-color: #1E3B70;
            border: none;
            padding: 10px 15px;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #29539B;
            transform: scale(1.02);
        }
        
        /* Custom styling for the sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f4f6f9 !important;
            border-right: 1px solid #e0e0e0 !important;
        }
        div[data-testid="stSidebarNav"] {
            background-color: transparent !important;
        }
        /* Sidebar link styling (Mimic agent-explorer buttons) */
        div[data-testid="stSidebarNav"] a {
            border-radius: 8px !important;
            margin: 8px 15px !important;
            padding: 10px 15px !important;
            transition: all 0.3s ease !important;
            text-decoration: none !important;
            display: block !important;
        }
        
        /* Force color on all nested elements (page names) */
        [data-testid="stPageLink-NavLink"] {
            background-color: #1E3B70 !important;
            border-radius: 8px !important;
            margin: 8px 0px !important;
            padding: 10px 15px !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stPageLink-NavLink"]:hover {
            background-color: #29539B !important;
            transform: scale(1.02) !important;
        }

        [data-testid="stPageLink-NavLink"] p,
        [data-testid="stPageLink-NavLink"] span {
            color: white !important;
            font-size: 16px !important;
            font-weight: bold !important;
        }
        
        /* Styling for code blocks */
        pre {
            background-color: #f7f7f7;
            border-radius: 5px;
            padding: 10px;
            overflow-x: auto;
        }
        
        code {
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar() -> None:
    """Настраивает боковое меню (sidebar) навигации приложения."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧬 GKT Oncology")
        st.markdown("Мультиагентная платформа<br>QLoRA Data Generation", unsafe_allow_html=True)
        
        st.markdown("---")
        st.page_link("main.py", label="Dashboard", icon="🏠")
        st.page_link("pages/1_Data_Pipeline.py", label="Data Pipeline", icon="📑")
        st.page_link("pages/4_Document_Storage.py", label="Document Storage", icon="📁")
        st.page_link("pages/2_QA_Viewer.py", label="QA Viewer", icon="🤖")
        st.page_link("pages/3_Sandbox.py", label="Sandbox", icon="🧪")
        st.page_link("pages/5_Evaluation.py", label="Evaluation", icon="📈")
        
        st.markdown("---")
        st.markdown("<small>dhc@sechenov.ai<br>Версия 0.1.0 MVP</small>", unsafe_allow_html=True)

def render_metric_card(title: str, value: str, icon: str = "") -> None:
    """
    Рендерит HTML-карточку для отображения метрики на дашборде.
    
    Args:
        title (str): Заголовок метрики (например, "Всего чанков").
        value (str): Числовое или текстовое значение метрики.
        icon (str, optional): Эмодзи-иконка для визуализации. Defaults to "".
    """
    st.markdown(f"""
    <div class="card">
        <div class="agent-icon">{icon}</div>
        <h3 style="margin-top:0;">{title}</h3>
        <p style="font-size: 2rem; font-weight: bold; color: #1E3B70; margin-bottom: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)
