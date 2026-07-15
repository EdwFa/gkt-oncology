import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_dataset():
    input_file = Path("data/qa_dataset/raw_dataset.jsonl")
    train_file = Path("data/training/train.jsonl")
    valid_file = Path("data/training/valid.jsonl")
    
    train_file.parent.mkdir(parents=True, exist_ok=True)
    
    valid_data = []
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Фильтруем забракованные
            if item.get("status") == "REJECTED":
                continue
                
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            
            if not question or not answer:
                continue
                
            # MLX-LM & Together AI формат: ChatML/Messages
            messages = [
                {"role": "system", "content": "Вы — AI-ассистент онколога. Отвечайте на вопросы точно и только на основе подтвержденных фактов. Если вы не знаете ответа или в контексте нет нужной информации, отвечайте строго 'НЕТ ДАННЫХ'."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            
            valid_data.append({"messages": messages})
            
    logger.info(f"Всего качественных QA-пар для обучения: {len(valid_data)}")
    
    # Перемешиваем и делим на train/valid (90/10)
    random.seed(42)
    random.shuffle(valid_data)
    
    split_idx = int(len(valid_data) * 0.9)
    train_data = valid_data[:split_idx]
    valid_data_split = valid_data[split_idx:]
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data_split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"Сохранено {len(train_data)} записей в {train_file}")
    logger.info(f"Сохранено {len(valid_data_split)} записей в {valid_file}")

if __name__ == "__main__":
    format_dataset()
