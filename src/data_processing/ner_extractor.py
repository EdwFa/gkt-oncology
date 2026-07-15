import os
import json
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

SYSTEM_PROMPT = """Вы - эксперт в медицинской онкологии ЖКТ.
Извлеките следующие сущности из предоставленного текста и верните их в формате JSON:
- "diagnoses": Диагнозы (с кодами МКБ, если есть)
- "drugs": Лекарственные препараты
- "procedures": Медицинские процедуры (хирургия, лучевая терапия и т.д.)
- "lab_tests": Лабораторные показатели
- "symptoms": Симптомы

Текст:
{text}

Верните только валидный JSON, без markdown-оболочек."""

async def extract_ner(session, chunk, sem, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = SYSTEM_PROMPT.format(text=chunk["text"])
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    
    async with sem:
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        entities = json.loads(content)
                    except:
                        entities = {"error": "Invalid JSON returned"}
                    
                    return {"chunk_id": chunk["chunk_id"], "entities": entities}
                else:
                    text = await response.text()
                    print(f"Error {response.status}: {text}")
                    return {"chunk_id": chunk["chunk_id"], "entities": {}}
        except Exception as e:
            print(f"Request failed: {e}")
            return {"chunk_id": chunk["chunk_id"], "entities": {}}

async def main():
    if not TOGETHER_API_KEY:
        print("Ошибка: TOGETHER_API_KEY не установлен в .env")
        return

    chunks_file = Path("data/chunks/all_chunks.jsonl")
    if not chunks_file.exists():
        print("Файл с чанками не найден. Сначала запустите chunker.py")
        return
        
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
            
    # Контролируем количество параллельных запросов к Together API
    sem = asyncio.Semaphore(20)
    
    async with aiohttp.ClientSession() as session:
        tasks = [extract_ner(session, chunk, sem) for chunk in chunks]
        results = await tqdm.gather(*tasks, desc="NER Extraction (Full)")
        
    output_dir = Path("data/processed_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "ner_pilot_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"NER извлечение завершено для {len(results)} чанков.")

if __name__ == "__main__":
    asyncio.run(main())
