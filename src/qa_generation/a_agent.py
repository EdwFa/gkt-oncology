import os
import json
import logging
import aiohttp
from typing import Dict, Any
from dotenv import load_dotenv

from .models import AnswerResponse, QuestionPrompt

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Вы - строгий медицинский эксперт по онкологии ЖКТ.
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
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    async def generate_answer(self, session: aiohttp.ClientSession, chunk_text: str, question: QuestionPrompt) -> AnswerResponse:
        prompt = SYSTEM_PROMPT.format(text=chunk_text, question=question.text)
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0, # Максимальная строгость и детерминированность
            "max_tokens": 1024,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=self.headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
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
