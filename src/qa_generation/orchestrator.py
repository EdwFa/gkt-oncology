import os
import json
import logging
import asyncio
import aiohttp
from pathlib import Path
from tqdm.asyncio import tqdm

from .q_agent import QuestionGenerator
from .a_agent import AnswerGenerator
from .ensemble_verifier import EnsembleVerifier
from .models import QAPair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_chunk(
    session: aiohttp.ClientSession,
    chunk: dict,
    q_agent: QuestionGenerator,
    a_agent: AnswerGenerator,
    verifier: EnsembleVerifier,
    pbar: tqdm,
    output_file: Path
):
    try:
        # Шаг 1: Генерация вопросов (Q-Agent)
        q_batch = await q_agent.generate_questions(session, chunk)
        if not q_batch or not q_batch.questions:
            return
        
        # Шаг 2: Генерация ответов (A-Agent) "вслепую"
        # Для начала можно запустить последовательно или параллельно батчем
        a_tasks = [
            a_agent.generate_answer(session, chunk.get("text", ""), q)
            for q in q_batch.questions
        ]
        a_responses = await asyncio.gather(*a_tasks)
        
        # Шаг 3: Собираем QA пары, отсеиваем отказы
        qa_pairs = []
        for q, a in zip(q_batch.questions, a_responses):
            qa_pair = QAPair(
                chunk_id=chunk.get("chunk_id", ""),
                question_id=q.id,
                question_type=q.type,
                question=q.text,
                answer=a.text,
                refusal=a.refusal
            )
            # Если ответа нет в тексте (refusal=True), то на этапе верификации мы её не прогоняем
            # В ТЗ указано: Negative examples для DPO. Сохраним их в сырой датасет, 
            # но не будем тратить ансамбль на проверку пустых ответов.
            if qa_pair.refusal:
                qa_pair.answer = "НЕТ ДАННЫХ"
            
            qa_pairs.append(qa_pair)
        
        # Шаг 4: Верификация ансамблем
        # Чтобы не упереться в лимиты, будем верифицировать последовательно в рамках одного чанка 
        # (параллелизм уже внутри верификатора на 4 модели)
        verified_pairs = []
        for qa in qa_pairs:
            if not qa.refusal:
                v_res = await verifier.verify_qa_pair(session, qa, chunk.get("text", ""))
                # Сохраняем в сырой датасет все пары, но добавляем метаданные верификации
                qa_dict = qa.model_dump()
                qa_dict["verification"] = v_res.model_dump()
                verified_pairs.append(qa_dict)
            else:
                qa_dict = qa.model_dump()
                qa_dict["verification"] = {"passed": False, "reason": "A-Agent refusal"}
                verified_pairs.append(qa_dict)
        
        # Шаг 5: Запись результатов
        with open(output_file, 'a', encoding='utf-8') as f:
            for pair in verified_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                
    except Exception as e:
        logger.error(f"Error processing chunk {chunk.get('chunk_id')}: {e}")
    finally:
        pbar.update(1)


async def main():
    chunks_path = Path("data/chunks/all_chunks.jsonl")
    ner_path = Path("data/processed_docs/ner_pilot_results.json")
    output_dir = Path("data/qa_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "raw_dataset.jsonl"
    
    if not chunks_path.exists() or not ner_path.exists():
        logger.error("Missing chunks or NER results files!")
        return
        
    # Сборка данных: объединяем чанки и NER сущности
    chunks = {}
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
                
    # Ограничиваемся только одним документом для отладки
    if chunks:
        first_doc_name = list(chunks.values())[0]["chunk_id"].split("_chunk_")[0]
        test_chunks = [c for c in chunks.values() if c["chunk_id"].startswith(first_doc_name)]
    else:
        test_chunks = []
    logger.info(f"Loaded {len(test_chunks)} chunks for processing (Document: {first_doc_name}).")
    
    # Очищаем файл перед запуском
    if output_file.exists():
        output_file.unlink()
        
    q_agent = QuestionGenerator()
    a_agent = AnswerGenerator()
    verifier = EnsembleVerifier()
    
    # Контролируем количество параллельных чанков (например, 2 чанка одновременно)
    # 2 чанка * 4 модели = 8 параллельных запросов на верификацию + 2 генерации
    sem = asyncio.Semaphore(2) 
    
    async def bound_process(session, chunk, pbar):
        async with sem:
            await process_chunk(session, chunk, q_agent, a_agent, verifier, pbar, output_file)

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        pbar = tqdm(total=len(test_chunks), desc="QA Pipeline")
        tasks = [bound_process(session, chunk, pbar) for chunk in test_chunks]
        await asyncio.gather(*tasks)
        pbar.close()
        
    logger.info("Pipeline completed!")

if __name__ == "__main__":
    asyncio.run(main())
