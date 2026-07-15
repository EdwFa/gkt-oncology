"""
Главный оркестратор QA-пайплайна.
Асинхронно управляет всем процессом: берет текстовые чанки, отдает их Q-Agent'у 
для генерации вопросов, затем передает вопросы A-Agent'у, собирает пары 
и отправляет их в EnsembleVerifier.
"""
import os
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from tqdm.asyncio import tqdm
from typing import Dict, Any, List

from .q_agent import QuestionGenerator
from .a_agent import AnswerGenerator
from .ensemble_verifier import EnsembleVerifier
from .models import QAPair, QuestionBatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_chunk(
    session: aiohttp.ClientSession,
    chunk: Dict[str, Any],
    q_agent: QuestionGenerator,
    a_agent: AnswerGenerator,
    verifier: EnsembleVerifier,
    pbar: tqdm,
    output_file: Path
) -> None:
    """
    Полный цикл обработки одного чанка: Q-генерация -> A-генерация -> Верификация.
    
    Args:
        session (aiohttp.ClientSession): HTTP сессия.
        chunk (Dict[str, Any]): Данные чанка текста.
        q_agent (QuestionGenerator): Инстанс Q-Agent'а.
        a_agent (AnswerGenerator): Инстанс A-Agent'а.
        verifier (EnsembleVerifier): Инстанс верификатора.
        pbar (tqdm): Индикатор прогресса.
        output_file (Path): Путь для дозаписи (append) готовых QA-пар.
    """
    try:
        # Шаг 1: Генерация вопросов (Q-Agent)
        q_batch: QuestionBatch = await q_agent.generate_questions(session, chunk)
        if not q_batch or not q_batch.questions:
            return
        
        # Шаг 2: Генерация ответов (A-Agent) "вслепую"
        # Запускаем ответы на все сгенерированные вопросы параллельно
        a_tasks = [
            a_agent.generate_answer(session, chunk.get("text", ""), q)
            for q in q_batch.questions
        ]
        a_responses = await asyncio.gather(*a_tasks)
        
        # Шаг 3: Формирование QA-пар и отсеивание "отказов" (refusals)
        qa_pairs: List[QAPair] = []
        for q, a in zip(q_batch.questions, a_responses):
            qa_pair = QAPair(
                chunk_id=chunk.get("chunk_id", ""),
                question_id=q.id,
                question_type=q.type,
                question=q.text,
                answer=a.text,
                refusal=a.refusal
            )
            # Если ответа нет в тексте (refusal=True), то мы не отдаем его ансамблю на оценку.
            # (Экономим токены). В ТЗ указано: такие пары пойдут для DPO-обучения.
            if qa_pair.refusal:
                qa_pair.answer = "НЕТ ДАННЫХ"
            
            qa_pairs.append(qa_pair)
        
        # Шаг 4: Верификация ансамблем
        # Оцениваем QA-пары последовательно внутри чанка (параллелизм уже есть внутри верификатора)
        verified_pairs: List[Dict[str, Any]] = []
        for qa in qa_pairs:
            if not qa.refusal:
                v_res = await verifier.verify_qa_pair(session, qa, chunk.get("text", ""))
                # Сохраняем в датасет все сгенерированные пары, добавляя метаданные верификации
                qa_dict = qa.model_dump()
                qa_dict["verification"] = v_res.model_dump()
                verified_pairs.append(qa_dict)
            else:
                # Пары с отказами сразу помечаются как "не прошедшие верификацию",
                # но сохраняются для DPO с причиной "A-Agent refusal"
                qa_dict = qa.model_dump()
                qa_dict["verification"] = {"passed": False, "reason": "A-Agent refusal"}
                verified_pairs.append(qa_dict)
        
        # Шаг 5: Запись результатов в файл (append)
        with open(output_file, 'a', encoding='utf-8') as f:
            for pair in verified_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                
    except Exception as e:
        logger.error(f"Ошибка при обработке чанка {chunk.get('chunk_id')}: {e}")
    finally:
        # В любом случае обновляем прогресс-бар
        pbar.update(1)


async def main() -> None:
    """
    Точка входа пайплайна генерации. Загружает чанки, NER-сущности и запускает 
    асинхронную обработку с контролем параллелизма (Semaphore).
    """
    chunks_path: Path = Path("data/chunks/all_chunks.jsonl")
    ner_path: Path = Path("data/processed_docs/ner_pilot_results.json")
    
    output_dir: Path = Path("data/qa_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file: Path = output_dir / "raw_dataset.jsonl"
    
    if not chunks_path.exists() or not ner_path.exists():
        logger.error("Отсутствуют необходимые файлы чанков или NER-экстракции!")
        return
        
    # Сборка данных: объединяем чанки и извлеченные из них NER сущности
    chunks: Dict[str, Dict[str, Any]] = {}
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
            
    with open(ner_path, 'r', encoding='utf-8') as f:
        ner_results = json.load(f)
        for ner in ner_results:
            c_id = ner["chunk_id"]
            if c_id in chunks:
                chunks[c_id]["entities"] = ner.get("entities", {})
                
    # ДЛЯ ОТЛАДКИ: Ограничиваемся только одним документом
    if chunks:
        first_doc_name = list(chunks.values())[0]["chunk_id"].split("_chunk_")[0]
        test_chunks = [c for c in chunks.values() if c["chunk_id"].startswith(first_doc_name)]
    else:
        test_chunks = []
        
    logger.info(f"Загружено {len(test_chunks)} чанков для обработки (Документ: {first_doc_name}).")
    
    # Очищаем файл перед свежим запуском
    if output_file.exists():
        output_file.unlink()
        
    # Инициализация агентов
    q_agent = QuestionGenerator()
    a_agent = AnswerGenerator()
    verifier = EnsembleVerifier()
    
    # Контролируем количество параллельных чанков (по умолчанию 2 чанка одновременно).
    # 2 чанка * 4 модели верификатора = 8 параллельных запросов + генераторы.
    sem = asyncio.Semaphore(2) 
    
    async def bound_process(session: aiohttp.ClientSession, chunk: Dict[str, Any], pbar: tqdm) -> None:
        async with sem:
            await process_chunk(session, chunk, q_agent, a_agent, verifier, pbar, output_file)

    # Устанавливаем ограничение TCP соединений
    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(test_chunks), desc="QA Pipeline")
        tasks = [bound_process(session, chunk, pbar) for chunk in test_chunks]
        await asyncio.gather(*tasks)
        pbar.close()
        
    logger.info("Пайплайн успешно завершен!")

if __name__ == "__main__":
    asyncio.run(main())
