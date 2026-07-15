import os
import json
import logging
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

from .models import QAPair, EnsembleVerification, VerificationScore

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

logger = logging.getLogger(__name__)

VERIFIER_PROMPT = """Вы — эксперт по онкологии ЖКТ. Оцените качество пары вопрос-ответ по шкале 1-10.

Критерии оценки:
- 9-10: Клинически корректно, полно, соответствует действующим КР
- 7-8: В основном корректно, допустимы мелкие неточности
- 5-6: Частично корректно, есть существенные пробелы
- 3-4: Существенные клинические ошибки
- 1-2: Клинически опасно, может привести к вреду пациенту

Вопрос: {question}
Ответ: {answer}
Контекст (раздел КР): {context}

Верните результат строго в формате JSON:
{{
  "score": 10.0,
  "reasoning": "Обоснование оценки (1-2 предложения)"
}}
"""

class EnsembleVerifier:
    def __init__(self):
        # Модели и их веса (будут калиброваться позже, пока базовые)
        self.models = [
            {"name": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "weight": 1.0},
            {"name": "Qwen/Qwen2.5-7B-Instruct-Turbo", "weight": 0.8},
            {"name": "google/gemma-2-27b-it", "weight": 0.9},
            {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "weight": 0.8}
        ]
        self.headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    async def _verify_with_model(self, session: aiohttp.ClientSession, qa_pair: QAPair, context: str, model_info: Dict) -> float:
        prompt = VERIFIER_PROMPT.format(
            question=qa_pair.question,
            answer=qa_pair.answer,
            context=context
        )
        
        payload = {
            "model": model_info["name"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=self.headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    try:
                        data = json.loads(content)
                        score_obj = VerificationScore(**data)
                        return score_obj.score
                    except Exception as e:
                        logger.error(f"Invalid JSON from {model_info['name']}: {e}")
                        return 0.0
                else:
                    return 0.0
        except Exception as e:
            logger.error(f"Verifier API error: {e}")
            return 0.0

    async def verify_qa_pair(self, session: aiohttp.ClientSession, qa_pair: QAPair, context: str) -> EnsembleVerification:
        tasks = [
            self._verify_with_model(session, qa_pair, context, model)
            for model in self.models
        ]
        
        # Запускаем все модели параллельно
        scores = await asyncio.gather(*tasks)
        
        valid_scores = []
        valid_weights = []
        model_scores_dict = {}
        
        for i, score in enumerate(scores):
            model_name = self.models[i]["name"]
            model_scores_dict[model_name] = score
            if score > 0:
                valid_scores.append(score)
                valid_weights.append(self.models[i]["weight"])
                
        if not valid_scores:
            return EnsembleVerification(
                s_final=0.0, consensus=0.0, confidence=0.0, disagreement=0.0, 
                model_scores=model_scores_dict, passed=False
            )
            
        # Расчет S_final (взвешенная сумма)
        s_final = np.average(valid_scores, weights=valid_weights)
        
        # Расчет Confidence (доля оценок >= 7)
        confidence = sum(1 for s in valid_scores if s >= 7.0) / len(valid_scores)
        
        # Расчет Disagreement (разброс)
        disagreement = max(valid_scores) - min(valid_scores)
        
        # Расчет Consensus
        epsilon = 0.1
        std_dev = np.std(valid_scores)
        consensus = 1.0 - (std_dev / (disagreement + epsilon))
        
        # Правила фильтрации (из ТЗ)
        passed = (s_final >= 7.0) and (consensus >= 0.6) and (confidence >= 0.67) and (disagreement <= 4.0)
        
        return EnsembleVerification(
            s_final=float(s_final),
            consensus=float(consensus),
            confidence=float(confidence),
            disagreement=float(disagreement),
            model_scores=model_scores_dict,
            passed=passed
        )
