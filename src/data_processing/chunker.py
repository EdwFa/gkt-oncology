import json
import logging
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_markdown(text: str, file_name: str) -> list[dict]:
    """Чанкирование с учетом структуры Markdown."""
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # Сначала режем по заголовкам, сохраняя иерархию
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(text)
    
    # Если какие-то секции получились слишком большими, аккуратно режем их по абзацам/предложениям
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=300,
        separators=["\n\n", "\n", r"(?<=\. )", r"(?<=\! )", r"(?<=\? )"],
        is_separator_regex=True
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    
    chunks = []
    for i, doc in enumerate(final_splits):
        # Восстанавливаем заголовки в метаданных для контекста (если нужно)
        metadata_str = " | ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
        chunk_text = doc.page_content
        
        # Если заголовки были удалены (strip_headers=True), мы могли бы вставить их обратно.
        # Но strip_headers=False оставляет их в тексте.
        
        chunks.append({
            "doc_name": file_name,
            "chunk_id": f"{file_name}_chunk_{i}",
            "text": chunk_text,
            "metadata": doc.metadata
        })
        
    return chunks

def process_parsed_docs():
    processed_dir = Path("data/processed_docs")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    parsed_files = list(processed_dir.glob("*.json"))
    # Исключаем файлы NER, если они там лежат
    parsed_files = [f for f in parsed_files if "ner_results" not in f.name and "ner_pilot_results" not in f.name]
    
    all_chunks = []
    
    for file_path in parsed_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            text = data.get("full_text", "")
            if not text:
                continue
                
            doc_chunks = chunk_markdown(text, data["file_name"])
            all_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error chunking {file_path}: {e}")
            
    output_file = chunks_dir / "all_chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    logger.info(f"Сформировано {len(all_chunks)} семантических чанков.")

if __name__ == "__main__":
    process_parsed_docs()

