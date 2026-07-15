"""
Модуль для парсинга PDF документов.
Извлекает текст с сохранением Markdown-структуры с помощью библиотеки Docling.
"""
import os
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional, List
from docling.document_converter import DocumentConverter

def extract_text_from_pdf(pdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает текст из PDF файла с сохранением разметки (Markdown) с помощью Docling.
    
    Args:
        pdf_path (str): Путь к исходному PDF-файлу.
        
    Returns:
        Optional[Dict[str, Any]]: Словарь с метаданными и извлеченным полным текстом,
                                  либо None в случае ошибки.
    """
    try:
        # Инициализация конвертера Docling
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Получаем полный текст в формате Markdown.
        # Docling отлично справляется с таблицами, списками и заголовками.
        full_text: str = result.document.export_to_markdown()
        
        return {
            "file_name": os.path.basename(pdf_path),
            "total_pages": -1, # docling абстрагирует постраничное разбиение
            "pages": [],
            "full_text": full_text
        }
    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")
        return None

def main() -> None:
    """
    Основная функция для потокового парсинга PDF файлов.
    Читает все файлы из директории data/raw_docs и сохраняет результаты
    в виде отдельных JSON файлов и единого all_parsed_docs.jsonl.
    """
    raw_docs_dir: Path = Path("data/raw_docs")
    processed_dir: Path = Path("data/processed_docs")
    
    # Создаем директорию для результатов, если ее нет
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Ищем все PDF документы в папке сырых данных
    pdf_files: List[Path] = list(raw_docs_dir.glob("*.pdf"))
    print(f"Найдено PDF файлов для парсинга: {len(pdf_files)}")
    
    all_parsed_docs: List[Dict[str, Any]] = []
    
    # Парсим все найденные документы с отображением прогресса
    for pdf_path in tqdm(pdf_files, desc="Парсинг PDF"):
        parsed_doc = extract_text_from_pdf(str(pdf_path))
        if parsed_doc:
            all_parsed_docs.append(parsed_doc)
            
            # Сохраняем каждый разобранный файл в отдельный JSON для удобства отладки
            output_file: Path = processed_dir / f"{parsed_doc['file_name']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc, f, ensure_ascii=False, indent=2)

    # Собираем общий JSONL файл со всеми текстами для передачи в Chunker
    combined_output: Path = processed_dir / "all_parsed_docs.jsonl"
    with open(combined_output, 'w', encoding='utf-8') as f:
        for doc in all_parsed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print(f"\nОбработка завершена. Успешно обработано: {len(all_parsed_docs)}/{len(pdf_files)}")

if __name__ == "__main__":
    main()
