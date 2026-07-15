# 🧬 GKT Oncology QLoRA & Multi-Agent Verification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Together AI](https://img.shields.io/badge/Together-AI-purple.svg)
![Apple MLX](https://img.shields.io/badge/Apple-MLX-black.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

Система поддержки принятия врачебных решений (СППВР) в области онкологии ЖКТ. Проект представляет собой полный цикл: от глубокого парсинга клинических рекомендаций до локального обучения специализированного LoRA-адаптера (Fine-Tuning) для LLM (база — Qwen2.5/Qwen3.5) с использованием **математической модели гетерогенного LLM-ансамбля** для автоматической верификации QA-датасетов.

## 🌟 Особенности проекта

- 📄 **Интеллектуальный парсинг PDF**: Использование инструмента `docling` для извлечения текстов с сохранением иерархии разметки и таблиц.
- 🧩 **Semantic Chunking**: Интеллектуальная нарезка текста (MarkdownHeaderTextSplitter + Regex) с полным сохранением структуры предложений и клинического контекста.
- 🤖 **Multi-Agent QA Pipeline**:
  - `Q-Agent`: генерирует 5 типов вопросов (фактологические, клинические сценарии, негативные и др.).
  - `A-Agent`: "слепой" генератор ответов по строгим правилам доказательной медицины.
- ⚖️ **Ensemble Verifier**: Ансамбль из нескольких передовых LLM (Llama-3, Qwen, Mixtral) независимо голосует за качество каждой пары (по шкале 1-10) и высчитывает агрегированный балл ($S_{final}$).
- 🚀 **Apple MLX Training**: Локальное квантованное дообучение (QLoRA) прямо на Mac (Apple Silicon) с минимальными требованиями к памяти.
- 📊 **Streamlit Dashboard**: Премиальный пользовательский интерфейс для управления данными, визуального просмотра чанков и тестирования сгенерированных вопросов.

## 🛠 Установка

1. **Клонирование репозитория:**
   ```bash
   git clone https://github.com/your-repo/gkt-oncology.git
   cd gkt-oncology
   ```

2. **Настройка виртуального окружения:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Окружение (API Ключи):**
   Создайте файл `.env` в корне проекта и добавьте ваш ключ Together AI (для генерации данных):
   ```ini
   TOGETHER_API_KEY="ваш_ключ"
   ```

4. **Установка Apple MLX (только для процессоров серии M1/M2/M3):**
   ```bash
   pip install mlx mlx-lm
   ```

## 🚀 Использование

### 1. Запуск Dashboard (Streamlit)
Управление всем пайплайном и просмотр результатов инкапсулированы в красивый UI:
```bash
streamlit run src/app/main.py
```

### 2. Подготовка данных (Data Pipeline)
Этап обработки клинических рекомендаций (консольный запуск):
- Парсинг сырых PDF из `data/raw_docs`:
  ```bash
  python src/data_processing/pdf_parser.py
  ```
- Семантическое чанкирование:
  ```bash
  python src/data_processing/chunker.py
  ```

### 3. Многоагентная генерация (QA Generation)
Асинхронный оркестратор, прогоняющий тексты через агентов и ансамбль верификаторов:
```bash
python -m src.qa_generation.orchestrator
```

### 4. Обучение (LoRA Fine-Tuning)
Форматирование датасета (ChatML) и старт локального обучения через фреймворк MLX:
```bash
python src/training/format_dataset.py
python src/training/train_lora.py
```

## 📂 Структура проекта

```text
├── data/
│   ├── raw_docs/           # Исходные PDF (Клинические рекомендации)
│   ├── processed_docs/     # Результаты парсинга (Markdown / JSON)
│   ├── chunks/             # Нарезанные семантические чанки
│   ├── qa_dataset/         # Сырые и верифицированные QA-пары
│   └── training/           # Датасеты для Fine-Tuning (train/valid.jsonl)
├── src/
│   ├── app/                # Streamlit Dashboard (UI и визуализация)
│   ├── data_processing/    # Скрипты парсинга, чанкирования и экстракции
│   ├── qa_generation/      # Архитектура агентов (Q, A) и Оркестратор
│   └── training/           # Скрипты форматирования и запуска LoRA
├── ТЗ.md                   # Подробное Техническое Задание и Мат. Аппарат
├── requirements.txt        # Список Python-зависимостей
└── README.md               # Документация (этот файл)
```

## 📋 Документация и архитектура
Подробное описание математической модели LLM-ансамбля, формулы доказательства надежности (теоремы о дисперсии ошибок) и подробные архитектурные решения описаны в файле [ТЗ.md](ТЗ.md).
