import os
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Пытаемся импортировать MLX
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """Вы — строгий судья-онколог. 
Ваша задача — оценить качество ответа, сгенерированного ИИ, по сравнению с эталонным ответом.

Оцените ответ ИИ от 1 до 5:
5 - Полностью верный и точный ответ.
4 - Верный ответ, но есть незначительные упущения.
3 - Частично верный, но не хватает важных деталей.
2 - В основном неверный или содержит галлюцинации.
1 - Полностью неверный или опасный ответ.

Вопрос: {question}
Эталонный ответ: {reference}
Ответ ИИ (Обученной модели): {prediction}

Верните результат строго в формате JSON:
{{
  "score": 5,
  "reasoning": "Обоснование..."
}}
"""

class LLMJudge:
    def __init__(self, model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"):
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
    async def evaluate_answer(self, session: aiohttp.ClientSession, question: str, reference: str, prediction: str) -> dict:
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
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
                        return {"score": data.get("score", 0), "reasoning": data.get("reasoning", "")}
                    except Exception as e:
                        logger.error(f"JSON Parse Error: {e}")
                        return {"score": 0, "reasoning": "Parse error"}
                else:
                    return {"score": 0, "reasoning": f"API Error {response.status}"}
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {"score": 0, "reasoning": "Request failed"}

async def run_evaluation():
    if not MLX_AVAILABLE:
        logger.error("Пакет mlx_lm не установлен. Для локальной генерации на Mac установите: pip install mlx-lm")
        return
        
    valid_file = Path("data/training/valid.jsonl")
    if not valid_file.exists():
        logger.error(f"Файл {valid_file} не найден.")
        return
        
    logger.info("Загрузка модели с LoRA адаптером...")
    try:
        # Загружаем базовую модель и адаптеры
        model, tokenizer = load("Qwen/Qwen2.5-7B-Instruct", adapter_path="adapters/")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели MLX: {e}")
        return
        
    valid_data = []
    with open(valid_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            valid_data.append(item)
            
    # Берем максимум 20 вопросов для быстрой оценки
    sample_size = min(20, len(valid_data))
    test_sample = valid_data[:sample_size]
    
    logger.info(f"Генерация ответов для {sample_size} вопросов...")
    
    results = []
    
    # 1. Генерация ответов нашей локальной моделью
    for item in tqdm(test_sample, desc="Генерация (MLX)"):
        messages = item["messages"]
        # Извлекаем системный промпт, вопрос и эталон
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        question = next((m["content"] for m in messages if m["role"] == "user"), "")
        reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        
        # Форматируем промпт для Qwen2.5
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question}
        ], tokenize=False, add_generation_prompt=True)
        
        # Генерация ответа локально
        prediction = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
        
        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction
        })
        
    # 2. Оценка ответов LLM Судьей (Qwen2.5 через API)
    logger.info("Оценка сгенерированных ответов LLM Судьей...")
    judge = LLMJudge(model="Qwen/Qwen2.5-7B-Instruct-Turbo")
    
    evaluated_results = []
    async with aiohttp.ClientSession() as session:
        for res in tqdm(results, desc="Судейство (API)"):
            eval_data = await judge.evaluate_answer(
                session, 
                question=res["question"], 
                reference=res["reference"], 
                prediction=res["prediction"]
            )
            res["score"] = eval_data["score"]
            res["reasoning"] = eval_data["reasoning"]
            evaluated_results.append(res)
            
    # 3. Подсчет статистики
    scores = [r["score"] for r in evaluated_results if r["score"] > 0]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    logger.info("=== Результаты Evaluation ===")
    logger.info(f"Средний балл (LLM-as-a-Judge Qwen2.5): {avg_score:.2f} / 5.0")
    
    # Сохраняем отчет
    report_file = Path("data/qa_dataset/evaluation_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "average_score": avg_score,
            "total_evaluated": len(scores),
            "details": evaluated_results
        }, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Подробный отчет сохранен в {report_file}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
