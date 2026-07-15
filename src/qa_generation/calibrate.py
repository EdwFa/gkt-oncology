import json
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calibrate_ensemble() -> None:
    """
    Анализирует сырые оценки (model_scores) из сырого датасета
    и вычисляет оптимальные веса для каждой модели ансамбля на основе её
    согласованности с медианой остальных моделей.
    """
    raw_dataset_path = Path("data/qa_dataset/raw_dataset.jsonl")
    
    if not raw_dataset_path.exists():
        logger.error(f"Файл датасета не найден: {raw_dataset_path}")
        return
        
    model_errors: Dict[str, List[float]] = {}
    
    with open(raw_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                verification = item.get("verification", {})
                scores = verification.get("model_scores", {})
                
                # Если у нас есть хотя бы 2 модели с валидными оценками (>0)
                valid_scores = {k: v for k, v in scores.items() if v > 0}
                
                if len(valid_scores) >= 2:
                    for model_name, score in valid_scores.items():
                        # Считаем медиану остальных моделей
                        other_scores = [v for k, v in valid_scores.items() if k != model_name]
                        
                        # Если осталась хотя бы 1 другая модель
                        if other_scores:
                            median_other = np.median(other_scores)
                            error = abs(score - median_other)
                            
                            if model_name not in model_errors:
                                model_errors[model_name] = []
                            model_errors[model_name].append(error)
                        
            except Exception as e:
                continue
                
    if not model_errors:
        logger.warning("Не найдено достаточного количества данных для калибровки.")
        return
        
    logger.info("=== Результаты калибровки ансамбля ===")
    
    # Считаем среднюю абсолютную ошибку (MAE) для каждой модели
    mean_errors: Dict[str, float] = {k: float(np.mean(v)) for k, v in model_errors.items()}
    
    for model, mae in mean_errors.items():
        logger.info(f"Модель: {model} | MAE от медианы остальных: {mae:.3f}")
        
    # Преобразуем ошибки в веса (обратно пропорционально)
    # Защита от деления на ноль
    epsilon = 0.01
    inverse_errors = {k: 1.0 / (v + epsilon) for k, v in mean_errors.items()}
    
    # Нормализуем так, чтобы максимальный вес был 1.0
    max_inv = max(inverse_errors.values())
    calibrated_weights = {k: round(v / max_inv, 2) for k, v in inverse_errors.items()}
    
    logger.info("\n=== Рекомендуемые новые веса ===")
    for model, weight in calibrated_weights.items():
        logger.info(f"'{model}': {weight}")
        
    # Сохраним в отдельный JSON для использования оркестратором,
    # либо выведем инструкцию для ручного обновления в коде.
    weights_file = Path("data/qa_dataset/calibrated_weights.json")
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(calibrated_weights, f, indent=4, ensure_ascii=False)
        
    logger.info(f"\nВеса сохранены в {weights_file}")

if __name__ == "__main__":
    calibrate_ensemble()
