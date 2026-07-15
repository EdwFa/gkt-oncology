"""
Модуль для извлечения медицинских сущностей (NER - Named Entity Recognition).
Отправляет чанки текста в LLM (Llama-3.3-70B) для извлечения диагнозов,
лекарств, процедур и симптомов в структурированном JSON-формате.
"""
import os
import json
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from typing import Dict, Any, List

# Загрузка переменных окружения (API-ключей)
load_dotenv()
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")

# Системный промпт, описывающий задачу для LLM
SYSTEM_PROMPT: str = """Вы - эксперт в медицинской онкологии ЖКТ.
Извлеките следующие сущности из предоставленного текста и верните их в формате JSON:
- "diagnoses": Диагнозы (с кодами МКБ, если есть)
- "drugs": Лекарственные препараты
- "procedures": Медицинские процедуры (хирургия, лучевая терапия и т.д.)
- "lab_tests": Лабораторные показатели
- "symptoms": Симптомы

Текст:
{text}

Верните только валидный JSON, без markdown-оболочек."""

async def extract_ner(
    session: aiohttp.ClientSession, 
    chunk: Dict[str, Any], 
    sem: asyncio.Semaphore, 
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
) -> Dict[str, Any]:
    """
    Асинхронно извлекает сущности из одного текстового чанка через API Together.
    
    Args:
        session (aiohttp.ClientSession): Сессия для HTTP-запросов.
        chunk (Dict[str, Any]): Словарь с данными чанка.
        sem (asyncio.Semaphore): Семафор для ограничения параллельных запросов.
        model (str): Название LLM-модели, используемой для экстракции.
        
    Returns:
        Dict[str, Any]: Словарь, содержащий ID чанка и извлеченные сущности (entities).
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Подставляем текст чанка в промпт
    prompt = SYSTEM_PROMPT.format(text=chunk["text"])
    
    # Формируем тело запроса
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # Низкая температура для стабильности извлечения фактов
        "max_tokens": 1024,
        "response_format": {"type": "json_object"} # Форсируем вывод в формате JSON
    }
    
    async with sem:
        try:
            # Выполняем асинхронный POST-запрос к Together API
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        entities: Dict[str, Any] = json.loads(content)
                    except json.JSONDecodeError:
                        entities = {"error": "Invalid JSON returned"}
                    
                    return {"chunk_id": chunk["chunk_id"], "entities": entities}
                else:
                    text = await response.text()
                    print(f"Ошибка API (Статус {response.status}): {text}")
                    return {"chunk_id": chunk["chunk_id"], "entities": {}}
        except Exception as e:
            print(f"Сетевая ошибка при обработке чанка {chunk['chunk_id']}: {e}")
            return {"chunk_id": chunk["chunk_id"], "entities": {}}

async def main() -> None:
    """
    Главная асинхронная функция. Загружает чанки из файла, 
    параллельно запускает извлечение NER и сохраняет результаты.
    """
    if not TOGETHER_API_KEY:
        print("Ошибка: TOGETHER_API_KEY не установлен в файле .env")
        return

    chunks_file: Path = Path("data/chunks/all_chunks.jsonl")
    if not chunks_file.exists():
        print("Файл с чанками не найден. Сначала запустите скрипт chunker.py")
        return
        
    chunks: List[Dict[str, Any]] = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
            
    # Контролируем количество параллельных запросов к Together API (ограничение до 20)
    sem = asyncio.Semaphore(20)
    
    async with aiohttp.ClientSession() as session:
        # Формируем список задач
        tasks = [extract_ner(session, chunk, sem) for chunk in chunks]
        # Запускаем задачи с отображением прогресс-бара
        results = await tqdm.gather(*tasks, desc="Извлечение NER")
        
    output_dir: Path = Path("data/processed_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем результаты в JSON-файл
    output_path = output_dir / "ner_pilot_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"Извлечение сущностей (NER) успешно завершено для {len(results)} чанков.")

if __name__ == "__main__":
    # Запуск асинхронного event loop'а
    asyncio.run(main())
