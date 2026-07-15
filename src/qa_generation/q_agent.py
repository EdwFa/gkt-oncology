import os
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any
from dotenv import load_dotenv

from .models import QuestionBatch

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Вы - строгий медицинский эксперт.
Ваша задача - сгенерировать обучающие вопросы (Q&A) СТРОГО на основе предоставленного текста.
Количество вопросов должно зависеть от объема фактической информации в тексте (обычно от 1 до 5 вопросов на фрагмент). Не придумывайте вопросы, ответов на которые нет в тексте!

Возможные типы вопросов (используйте только те, для которых ЕСТЬ ДАННЫЕ в тексте):
1. factual: вопросы на конкретные факты, определения, дозировки, коды МКБ.
2. clinical_scenario: клинические случаи (только если в тексте описаны критерии диагностики/лечения).
3. calculation: расчетные вопросы (только если в тексте есть формулы или точные дозы).
4. comparative: сравнение подходов (только если в тексте есть сравнение).
5. reasoning_chain: логические рассуждения на основе алгоритмов из текста.
6. negative: вопросы, связанные с текстом, но на которые в нем НЕТ прямого ответа (чтобы обучить модель отказам - около 10% вопросов).

ПРАВИЛА:
1. Вопрос должен быть полностью понятен без контекста (указывайте названия препаратов, болезней). Используйте извлеченные сущности: {entities}.
2. НЕ генерируйте расчетные или клинические вопросы, если в тексте нет для них основы.
3. Верните результат строго в формате JSON, соответствующем следующей схеме:
{{
  "questions": [
    {{"id": "уникальный_id_1", "type": "тип_вопроса", "text": "Текст вопроса"}},
    ...
  ]
}}

Текст клинической рекомендации:
{text}
"""

class QuestionGenerator:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    async def generate_questions(self, session: aiohttp.ClientSession, chunk: Dict[str, Any]) -> QuestionBatch:
        text = chunk.get("text", "")
        entities = json.dumps(chunk.get("entities", {}), ensure_ascii=False, indent=2)
        
        prompt = SYSTEM_PROMPT.format(text=text, entities=entities)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3, # Немного креативности для генерации разных типов вопросов
            "max_tokens": 4096,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=self.headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        data = json.loads(content)
                        return QuestionBatch(**data)
                    except Exception as e:
                        logger.error(f"Failed to parse Q-Agent response for chunk {chunk.get('chunk_id')}: {e}")
                        return QuestionBatch(questions=[])
                else:
                    text_resp = await response.text()
                    logger.error(f"Q-Agent API Error {response.status}: {text_resp}")
                    return QuestionBatch(questions=[])
        except Exception as e:
            logger.error(f"Q-Agent request failed: {e}")
            return QuestionBatch(questions=[])
