"""
Модуль A-Agent (Генератор ответов).
"Слепой" агент, который отвечает на сгенерированные вопросы, строго опираясь на текст.
Отвечает за формирование "refusals" (отказов), если данных в тексте недостаточно.
"""
import os
import json
import logging
import aiohttp
from typing import Dict, Any
from dotenv import load_dotenv

from .models import AnswerResponse, QuestionPrompt

load_dotenv()
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")

logger = logging.getLogger(__name__)

# Промпт, требующий строгой фактологии
SYSTEM_PROMPT: str = """Вы - строгий медицинский эксперт по онкологии ЖКТ.
Ваша задача - ответить на вопрос ИСКЛЮЧИТЕЛЬНО используя предоставленный текст клинической рекомендации.

Текст клинической рекомендации:
{text}

Вопрос:
{question}

ПРАВИЛА:
1. Если ответ на вопрос ЕСТЬ в тексте, дайте точный, развернутый и клинически корректный ответ.
2. Если ответа на вопрос НЕТ в тексте (даже если вы знаете ответ из других источников), вы ОБЯЗАНЫ вернуть в поле text "НЕТ ДАННЫХ" и установить refusal=true.
3. Верните результат строго в формате JSON:
{{
  "text": "ваш ответ или НЕТ ДАННЫХ",
  "refusal": false
}}
"""

class AnswerGenerator:
    """Класс генератора ответов через API Together."""
    
    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """Инициализация генератора."""
        self.model: str = model
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    async def generate_answer(self, session: aiohttp.ClientSession, chunk_text: str, question: QuestionPrompt) -> AnswerResponse:
        """
        Асинхронно генерирует ответ на один вопрос.
        
        Args:
            session (aiohttp.ClientSession): HTTP сессия.
            chunk_text (str): Текст чанка, на который должен опираться ответ.
            question (QuestionPrompt): Pydantic модель вопроса от Q-Agent.
            
        Returns:
            AnswerResponse: Pydantic модель ответа (содержит текст ответа и флаг refusal).
        """
        prompt: str = SYSTEM_PROMPT.format(text=chunk_text, question=question.text)
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0, # Максимальная строгость и детерминированность (без креатива)
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=self.headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content: str = result["choices"][0]["message"]["content"]
                    try:
                        data = json.loads(content)
                        return AnswerResponse(**data)
                    except Exception as e:
                        logger.error(f"Failed to parse A-Agent response for question {question.id}: {e}")
                        return AnswerResponse(text="НЕТ ДАННЫХ", refusal=True)
                else:
                    text_resp = await response.text()
                    logger.error(f"A-Agent API Error {response.status}: {text_resp}")
                    return AnswerResponse(text="НЕТ ДАННЫХ", refusal=True)
        except Exception as e:
            logger.error(f"A-Agent request failed: {e}")
            return AnswerResponse(text="НЕТ ДАННЫХ", refusal=True)
