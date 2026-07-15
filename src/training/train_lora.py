import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_mlx_lora(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    Запускает локальную тренировку LoRA через Apple MLX Framework.
    Настройки подобраны для отладки на Mac: минимальный батч и малое количество итераций.
    """
    data_dir = "data/training"
    
    if not Path(f"{data_dir}/train.jsonl").exists():
        logger.error(f"Файл {data_dir}/train.jsonl не найден!")
        return

    # Команда для запуска mlx_lm
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--iters", "50",           # 50 итераций для быстрого теста
        "--batch-size", "1",       # 1 для экономии RAM/VRAM
        "--num-layers", "16",      # Тренируем только 16 слоев
        "--learning-rate", "1e-4",
        "--save-every", "50"       # Сохраним адаптер в конце
    ]
    
    logger.info(f"Запуск MLX LoRA: {' '.join(cmd)}")
    
    try:
        # Запускаем как дочерний процесс, вывод будет идти прямо в консоль
        subprocess.run(cmd, check=True)
        logger.info("Тренировка успешно завершена! Адаптеры сохранены в папку 'adapters'.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка во время тренировки: {e}")

if __name__ == "__main__":
    train_mlx_lora()
