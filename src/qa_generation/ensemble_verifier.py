"""
Модуль Ensemble Verifier (Ансамблевый верификатор).
Имплементирует математическую модель из ТЗ для оценки качества QA-пар.
Использует гетерогенный ансамбль из 4 LLM для расчета S_final, Consensus и Confidence.
"""
import os
import json
import logging
import asyncio
import aiohttp
import numpy as np
from typing import Dict, List, Any
from dotenv import load_dotenv

from .models import QAPair, EnsembleVerification, VerificationScore

load_dotenv()
TOGETHER_API_KEY: str | None = os.getenv("TOGETHER_API_KEY")

logger = logging.getLogger(__name__)

# Промпт для ансамбля (оценивает ответ по 10-балльной шкале на основе контекста)
VERIFIER_PROMPT: str = """Вы — эксперт по онкологии ЖКТ. Оцените качество пары вопрос-ответ по шкале 1-10.

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
    """
    Класс верификатора, который отправляет QA-пару нескольким моделям
    и агрегирует их оценки по заданным математическим формулам.
    """
    
    def __init__(self) -> None:
        # Модели и их веса (будут калиброваться позже на основе экспертных оценок врачей)
        self.models: List[Dict[str, Any]] = [
            {"name": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "weight": 1.0},
            {"name": "Qwen/Qwen2.5-7B-Instruct-Turbo", "weight": 0.8},
            {"name": "google/gemma-2-27b-it", "weight": 0.9},
            {"name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "weight": 0.8}
        ]
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

    async def _verify_with_model(
        self, 
        session: aiohttp.ClientSession, 
        qa_pair: QAPair, 
        context: str, 
        model_info: Dict[str, Any]
    ) -> float:
        """
        Получает оценку (score) от одной конкретной модели из ансамбля.
        
        Args:
            session (aiohttp.ClientSession): HTTP сессия.
            qa_pair (QAPair): Оцениваемая QA-пара.
            context (str): Исходный текст из КР.
            model_info (Dict[str, Any]): Словарь с именем и весом модели.
            
        Returns:
            float: Оценка от 1 до 10 (или 0.0 в случае ошибки).
        """
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
            "temperature": 0.0, # Строго детерминированная оценка
            "max_tokens": 512,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.together.xyz/v1/chat/completions", headers=self.headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content: str = result["choices"][0]["message"]["content"]
                    try:
                        data = json.loads(content)
                        score_obj = VerificationScore(**data)
                        return float(score_obj.score)
                    except Exception as e:
                        logger.error(f"Invalid JSON from {model_info['name']}: {e}")
                        return 0.0
                else:
                    return 0.0
        except Exception as e:
            logger.error(f"Verifier API error: {e}")
            return 0.0

    async def verify_qa_pair(self, session: aiohttp.ClientSession, qa_pair: QAPair, context: str) -> EnsembleVerification:
        """
        Прогоняет QA-пару через весь ансамбль моделей, вычисляет статистику
        и принимает решение о прохождении проверки.
        
        Args:
            session (aiohttp.ClientSession): HTTP сессия.
            qa_pair (QAPair): Оцениваемая QA-пара.
            context (str): Исходный текст из КР.
            
        Returns:
            EnsembleVerification: Итоговый результат верификации с метриками.
        """
        # Формируем список задач для параллельного запроса ко всем 4 моделям
        tasks = [
            self._verify_with_model(session, qa_pair, context, model)
            for model in self.models
        ]
        
        # Запускаем все модели параллельно
        scores: List[float] = await asyncio.gather(*tasks)
        
        valid_scores: List[float] = []
        valid_weights: List[float] = []
        model_scores_dict: Dict[str, float] = {}
        
        # Фильтруем успешные оценки (в случае API-ошибок score = 0.0)
        for i, score in enumerate(scores):
            model_name = self.models[i]["name"]
            model_scores_dict[model_name] = score
            if score > 0:
                valid_scores.append(score)
                valid_weights.append(self.models[i]["weight"])
                
        if not valid_scores:
            # Если все модели упали с ошибкой
            return EnsembleVerification(
                s_final=0.0, consensus=0.0, confidence=0.0, disagreement=0.0, 
                model_scores=model_scores_dict, passed=False
            )
            
        # Математика из ТЗ: Расчет S_final (взвешенное среднее)
        s_final = float(np.average(valid_scores, weights=valid_weights))
        
        # Расчет Confidence (доля оценок >= 7.0)
        confidence = float(sum(1 for s in valid_scores if s >= 7.0) / len(valid_scores))
        
        # Расчет Disagreement (размах оценок)
        disagreement = float(max(valid_scores) - min(valid_scores))
        
        # Расчет Consensus (учет стандартного отклонения и разброса)
        epsilon = 0.1
        std_dev = np.std(valid_scores)
        consensus = float(1.0 - (std_dev / (disagreement + epsilon)))
        
        # Строгие правила фильтрации
        # S_final >= 7.0, Consensus >= 0.6, Confidence >= 0.67, разброс <= 4 баллов
        passed = (s_final >= 7.0) and (consensus >= 0.6) and (confidence >= 0.67) and (disagreement <= 4.0)
        
        return EnsembleVerification(
            s_final=s_final,
            consensus=consensus,
            confidence=confidence,
            disagreement=disagreement,
            model_scores=model_scores_dict,
            passed=passed
        )
