from pydantic import BaseModel, Field
from typing import List, Optional, Literal

QuestionType = Literal[
    "factual", 
    "clinical_scenario", 
    "calculation", 
    "comparative", 
    "negative", 
    "adversarial", 
    "reasoning_chain"
]

class QuestionPrompt(BaseModel):
    id: str = Field(description="Уникальный идентификатор вопроса в рамках чанка")
    type: QuestionType = Field(description="Тип вопроса согласно ТЗ")
    text: str = Field(description="Текст сгенерированного вопроса")

class QuestionBatch(BaseModel):
    questions: List[QuestionPrompt] = Field(description="Сгенерированный список вопросов для конкретного чанка")

class AnswerResponse(BaseModel):
    text: str = Field(description="Текст ответа. Если информации нет в контексте, должен содержать 'НЕТ ДАННЫХ'")
    refusal: bool = Field(default=False, description="Установлен в True, если агент отказался отвечать (т.е. нет данных)")

class QAPair(BaseModel):
    chunk_id: str
    question_id: str
    question_type: QuestionType
    question: str
    answer: str
    refusal: bool

class VerificationScore(BaseModel):
    score: float = Field(description="Оценка от 1 до 10")
    reasoning: str = Field(description="Обоснование оценки (1-2 предложения)")

class EnsembleVerification(BaseModel):
    s_final: float
    consensus: float
    confidence: float
    disagreement: float
    model_scores: dict[str, float]
    passed: bool
