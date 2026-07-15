"""
Модуль для семантического чанкирования (нарезки) текста.
Использует заголовки Markdown для сохранения иерархии и регулярные выражения
для безопасного разбиения текста по границам предложений.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_markdown(text: str, file_name: str) -> List[Dict[str, Any]]:
    """
    Разбивает Markdown текст на смысловые фрагменты (чанки) с учетом иерархии заголовков.
    
    Args:
        text (str): Полный текст документа в формате Markdown.
        file_name (str): Имя исходного файла (для метаданных).
        
    Returns:
        List[Dict[str, Any]]: Список словарей, содержащих текст чанка и его метаданные (заголовки).
    """
    # Определяем уровни заголовков для первичной нарезки
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # 1. Сначала режем по заголовкам, сохраняя саму иерархию (strip_headers=False)
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(text)
    
    # 2. Если внутри раздела текст слишком большой (например, длинная таблица или абзац),
    # дополнительно режем его по абзацам или границам предложений.
    # Используем lookbehind regex (например, `(?<=\. )`), чтобы точка оставалась в конце предложения.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,           # Максимальный размер чанка в символах
        chunk_overlap=300,         # Перекрытие между чанками для сохранения контекста
        separators=["\n\n", "\n", r"(?<=\. )", r"(?<=\! )", r"(?<=\? )"],
        is_separator_regex=True
    )
    
    # Применяем вторичное чанкирование
    final_splits = text_splitter.split_documents(md_header_splits)
    
    chunks: List[Dict[str, Any]] = []
    for i, doc in enumerate(final_splits):
        chunk_text: str = doc.page_content
        
        # Формируем итоговый объект чанка
        chunks.append({
            "doc_name": file_name,
            "chunk_id": f"{file_name}_chunk_{i}",
            "text": chunk_text,
            "metadata": doc.metadata
        })
        
    return chunks

def process_parsed_docs() -> None:
    """
    Основная функция для обработки всех спарсенных документов.
    Читает JSON файлы из data/processed_docs, режет их на чанки 
    и сохраняет в единый файл all_chunks.jsonl.
    """
    processed_dir: Path = Path("data/processed_docs")
    chunks_dir: Path = Path("data/chunks")
    
    # Создаем папку для чанков, если ее нет
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем все разобранные документы, исключая файлы с результатами NER
    parsed_files: List[Path] = list(processed_dir.glob("*.json"))
    parsed_files = [f for f in parsed_files if "ner_results" not in f.name and "ner_pilot_results" not in f.name]
    
    all_chunks: List[Dict[str, Any]] = []
    
    for file_path in parsed_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data: Dict[str, Any] = json.load(f)
                
            text: str = data.get("full_text", "")
            if not text:
                continue
                
            # Запускаем чанкирование для конкретного документа
            doc_chunks = chunk_markdown(text, data["file_name"])
            all_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Ошибка при чанкировании {file_path}: {e}")
            
    # Сохраняем все чанки в один JSONL файл
    output_file: Path = chunks_dir / "all_chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logger.info(f"Сформировано {len(all_chunks)} семантических чанков.")

if __name__ == "__main__":
    process_parsed_docs()
