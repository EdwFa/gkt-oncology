"""
Pydantic модели для строгой типизации данных в пайплайне QA-генерации.
Определяют форматы вопросов, ответов, оценок и структуры итогового датасета.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

# Возможные типы вопросов согласно Техническому Заданию
QuestionType = Literal[
    "factual",           # Фактологические вопросы
    "clinical_scenario", # Клинические сценарии (кейсы)
    "calculation",       # Расчетные вопросы (дозировки)
    "comparative",       # Сравнительные вопросы
    "negative",          # Негативные вопросы (нет ответа в тексте)
    "adversarial",       # Провокационные вопросы (содержат подвох)
    "reasoning_chain"    # Логические цепочки
]

class QuestionPrompt(BaseModel):
    """Схема отдельного сгенерированного вопроса (от Q-Agent)"""
    id: str = Field(description="Уникальный идентификатор вопроса в рамках чанка")
    type: QuestionType = Field(description="Тип вопроса согласно ТЗ")
    text: str = Field(description="Текст сгенерированного вопроса")

class QuestionBatch(BaseModel):
    """Батч вопросов, генерируемый Q-Agent'ом на один текстовый чанк"""
    questions: List[QuestionPrompt] = Field(description="Сгенерированный список вопросов для конкретного чанка")

class AnswerResponse(BaseModel):
    """Схема ответа, сгенерированного A-Agent'ом"""
    text: str = Field(description="Текст ответа. Если информации нет в контексте, должен содержать 'НЕТ ДАННЫХ'")
    refusal: bool = Field(default=False, description="Установлен в True, если агент отказался отвечать (т.е. нет данных)")

class QAPair(BaseModel):
    """Полная QA-пара, готовая для передачи верификатору"""
    chunk_id: str
    question_id: str
    question_type: QuestionType
    question: str
    answer: str
    refusal: bool

class VerificationScore(BaseModel):
    """Оценка одной LLM из ансамбля верификаторов"""
    score: float = Field(description="Оценка от 1 до 10")
    reasoning: str = Field(description="Обоснование оценки (1-2 предложения)")

class EnsembleVerification(BaseModel):
    """Итоговый результат верификации QA-пары всем ансамблем"""
    s_final: float = Field(description="Итоговый взвешенный балл ансамбля")
    consensus: float = Field(description="Уровень консенсуса между моделями")
    confidence: float = Field(description="Уверенность (доля оценок >= 7)")
    disagreement: float = Field(description="Разброс (max - min оценка)")
    model_scores: Dict[str, float] = Field(description="Сырые оценки каждой модели")
    passed: bool = Field(description="Прошла ли пара строгие критерии отбора")
