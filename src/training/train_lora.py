"""
Скрипт для запуска локального обучения адаптеров (LoRA) на Apple Silicon (Mac) 
с использованием фреймворка mlx_lm. 
Позволяет эффективно файнтюнить LLM на процессорах M1/M2/M3 без внешних GPU.
"""
import subprocess
import sys
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_mlx_lora(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    """
    Запускает локальную тренировку LoRA через Apple MLX Framework.
    Настройки подобраны для отладки на Mac: минимальный батч и малое количество итераций.
    
    Args:
        model_name (str): HuggingFace ID базовой модели для дообучения.
    """
    data_dir: str = "data/training"
    
    if not Path(f"{data_dir}/train.jsonl").exists():
        logger.error(f"Файл {data_dir}/train.jsonl не найден! Сначала сгенерируйте и отформатируйте датасет.")
        return

    # Формирование команды для запуска модуля mlx_lm.lora
    cmd: List[str] = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--iters", "50",           # 50 итераций для быстрого теста (в проде нужно 1000+)
        "--batch-size", "1",       # 1 для экономии RAM/VRAM на Mac
        "--num-layers", "16",      # Тренируем только 16 слоев для экономии памяти
        "--learning-rate", "1e-4", # Стандартный learning rate для LoRA
        "--save-every", "50"       # Сохраним адаптер в конце
    ]
    
    logger.info(f"Запуск MLX LoRA: {' '.join(cmd)}")
    
    try:
        # Запускаем как дочерний процесс. check=True выбросит исключение при ошибке (код возврата != 0).
        subprocess.run(cmd, check=True)
        logger.info("Тренировка успешно завершена! Адаптеры сохранены в папку 'adapters'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка во время тренировки (код возврата {e.returncode}): {e}")

if __name__ == "__main__":
    train_mlx_lora()
