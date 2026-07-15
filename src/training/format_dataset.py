"""
Скрипт для форматирования датасета.
Берет сырой сгенерированный датасет, фильтрует по качеству (отбрасывает не прошедшие верификацию)
и конвертирует в формат ChatML/Messages, требуемый для обучения моделей (в т.ч. MLX-LM).
"""
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_dataset() -> None:
    """
    Основная функция форматирования и разделения датасета.
    Читает raw_dataset.jsonl, фильтрует плохие пары, формирует структуру prompt-completion
    и разбивает на train (90%) и valid (10%).
    """
    input_file: Path = Path("data/qa_dataset/raw_dataset.jsonl")
    train_file: Path = Path("data/training/train.jsonl")
    valid_file: Path = Path("data/training/valid.jsonl")
    
    # Создаем директорию для обучающих данных, если ее нет
    train_file.parent.mkdir(parents=True, exist_ok=True)
    
    valid_data: List[Dict[str, Any]] = []
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item: Dict[str, Any] = json.loads(line)
            # Фильтруем забракованные (проверка по ключу passed в словаре verification)
            verification: Dict[str, Any] = item.get("verification", {})
            if verification.get("passed") is False:
                continue
            
            # Обратная совместимость: если есть статус REJECTED
            if item.get("status") == "REJECTED":
                continue
                
            question: str = item.get("question", "").strip()
            answer: str = item.get("answer", "").strip()
            
            if not question or not answer:
                continue
                
            # Формирование формата MLX-LM & Together AI: ChatML/Messages
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": "Вы — AI-ассистент онколога. Отвечайте на вопросы точно и только на основе подтвержденных фактов. Если вы не знаете ответа или в контексте нет нужной информации, отвечайте строго 'НЕТ ДАННЫХ'."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            valid_data.append({"messages": messages})
            
    logger.info(f"Всего качественных QA-пар для обучения: {len(valid_data)}")
    
    if len(valid_data) == 0:
        logger.warning("Нет данных для обучения после фильтрации!")
        return
        
    # Перемешиваем датасет для случайной выборки
    random.seed(42)
    random.shuffle(valid_data)
    
    # Разделение на train/valid (90/10)
    split_idx: int = int(len(valid_data) * 0.9)
    train_data: List[Dict[str, Any]] = valid_data[:split_idx]
    valid_data_split: List[Dict[str, Any]] = valid_data[split_idx:]
    
    # Запись тренировочного сета
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Запись валидационного сета
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Сохранено {len(train_data)} записей в {train_file}")
    logger.info(f"Сохранено {len(valid_data_split)} записей в {valid_file}")

if __name__ == "__main__":
    format_dataset()
