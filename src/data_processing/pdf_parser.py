import os
import json
from tqdm import tqdm
from pathlib import Path
from docling.document_converter import DocumentConverter

def extract_text_from_pdf(pdf_path: str) -> dict:
    """Извлекает текст из PDF файла с помощью Docling."""
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Получаем полный текст в формате Markdown, который Docling генерирует очень хорошо
        full_text = result.document.export_to_markdown()
        
        return {
            "file_name": os.path.basename(pdf_path),
            "total_pages": -1, # docling-markdown abstract this away
            "pages": [],
            "full_text": full_text
        }
    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")
        return None

def main():
    raw_docs_dir = Path("data/raw_docs")
    processed_dir = Path("data/processed_docs")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(raw_docs_dir.glob("*.pdf"))
    
    # Парсим все найденные документы
    
    print(f"Найдено PDF файлов для парсинга: {len(pdf_files)}")
    
    all_parsed_docs = []
    
    for pdf_path in tqdm(pdf_files, desc="Парсинг PDF"):
        parsed_doc = extract_text_from_pdf(str(pdf_path))
        if parsed_doc:
            all_parsed_docs.append(parsed_doc)
            
            # Опционально сохраним каждый файл отдельно
            output_file = processed_dir / f"{parsed_doc['file_name']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_doc, f, ensure_ascii=False, indent=2)

    # Общий JSONL (опционально, если не будет слишком большим)
    combined_output = processed_dir / "all_parsed_docs.jsonl"
    with open(combined_output, 'w', encoding='utf-8') as f:
        for doc in all_parsed_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            
    print(f"\nОбработка завершена. Успешно обработано: {len(all_parsed_docs)}/{len(pdf_files)}")

if __name__ == "__main__":
    main()
