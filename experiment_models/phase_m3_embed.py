# phase_m3_embed.py
# Фаза M3: эмбеддинги выборки одной моделью. Поддерживает шардирование
# на 2 GPU (как phase_x2 в experiment_xml): два процесса, шард по id %% 2.
#
# В предыдущем эксперименте один из двух процессов «быстро падал после
# начала» — логов не сохранилось. Здесь:
#   - лог КАЖДОГО процесса пишется в logs/m3_{model}_shard{N}.log
#     (print + файл), включая полный traceback при падении;
#   - faulthandler ловит hard-crash (segfault/abort) в тот же файл;
#   - свободная VRAM печатается до загрузки модели;
#   - CUDA OOM → автоматическое уменьшение батча (в em_model_loader).
#
# Запуск (два процесса, по одному на GPU):
#   python phase_m3_embed.py --model e5_large --device cuda:0 --shards 2 --shard-id 0
#   python phase_m3_embed.py --model e5_large --device cuda:1 --shards 2 --shard-id 1
#
# Выход:
#   embeddings/{model}__shard{N}.npy       (M, dim) float32
#   embeddings/{model}__shard{N}_ids.npy   (M,) int64 — id документов
#   embeddings/{model}__shard{N}_meta.json (время, док/с, peak VRAM)

import argparse
import faulthandler
import io
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from em_config import EMB_DIR, LOG_DIR, SAMPLE_CSV  # noqa: E402


def _setup_logging(model_key: str, shard_id: int, tag: str = ''):
    """print → консоль + файл лога шарда. Возвращает (logger, log_file)."""
    LOG_DIR.mkdir(exist_ok=True)
    suffix = f"_{tag}" if tag else ''
    log_path = LOG_DIR / f"m3_{model_key}{suffix}_shard{shard_id}.log"
    fh = io.open(log_path, 'w', encoding='utf-8', errors='replace')
    faulthandler.enable(fh)

    def logger(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        fh.write(line + '\n')
        fh.flush()

    return logger, fh, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard-id', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='отладка: первые N док.')
    ap.add_argument('--tag', default='', help='суффикс выходных файлов для отдельного прогона')
    ap.add_argument('--random-sample', action='store_true', help='случайная выборка вместо первых N док.')
    ap.add_argument('--seed', type=int, default=20260818, help='seed случайной выборки')
    ap.add_argument('--sample', type=str, default=str(SAMPLE_CSV), help='CSV с колонкой subject_text или text')
    ap.add_argument('--max-length', type=int, default=0, help='единый лимит токенов для сравнения')
    ap.add_argument('--batch-size', type=int, default=0, help='переопределение размера батча')
    args = ap.parse_args()

    logger, log_fh, log_path = _setup_logging(args.model, args.shard_id, args.tag)
    try:
        logger(f"=== m3 embed: model={args.model} device={args.device} "
               f"shard {args.shard_id}/{args.shards} limit={args.limit} ===")

        df = pd.read_csv(args.sample, encoding='utf-8')
        if 'subject_text' not in df and 'text' in df:
            df['subject_text'] = df['text']
        df['subject_text'] = df['subject_text'].fillna('')
        df = df[(df['id'] % args.shards) == args.shard_id].reset_index(drop=True)
        if args.limit:
            if args.random_sample:
                df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
            else:
                df = df.head(args.limit)
        logger(f"sample: {len(df)} документов (шард {args.shard_id})")
        if len(df) == 0:
            logger("нечего делать")
            return

        # Импорт ПОСЛЕ настройки лога, чтобы traceback падал в файл
        from em_model_loader import EmbeddingModel

        t0 = time.time()
        with EmbeddingModel(args.model, device=args.device, logger=logger) as em:
            if args.max_length:
                em.meta['max_len'] = args.max_length
            if args.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(args.device)
            embs = em.encode(
                df['subject_text'].tolist(),
                batch_size=args.batch_size or None,
            )
        dt = time.time() - t0

        EMB_DIR.mkdir(exist_ok=True)
        suffix = f"__{args.tag}" if args.tag else ''
        stem = EMB_DIR / f"{args.model}{suffix}__shard{args.shard_id}"
        np.save(f"{stem}.npy", embs)
        np.save(f"{stem}_ids.npy", df['id'].to_numpy(dtype=np.int64))
        peak_gb = (torch.cuda.max_memory_allocated(args.device) / 1024**3
                   if args.device.startswith('cuda') else 0.0)
        meta = {
            'model': args.model, 'device': args.device,
            'shard_id': args.shard_id, 'shards': args.shards,
            'n_docs': int(len(df)), 'elapsed_s': round(dt, 1),
            'docs_per_s': round(len(df) / max(dt, 1e-9), 1),
            'avg_chars': round(float(df['subject_text'].str.len().mean()), 1),
            'avg_words': round(float(df['subject_text'].str.split().str.len().mean()), 1),
            'source_composition': (
                df['source_group'].value_counts().astype(int).to_dict()
                if 'source_group' in df else {'XML subject_text': int(len(df))}
            ),
            'peak_vram_gb': round(peak_gb, 2),
        }
        with io.open(f"{stem}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger(f"DONE {len(df)} док. за {dt:.0f} с ({meta['docs_per_s']} док/с, "
               f"peak VRAM {peak_gb:.1f} GB)")
    except Exception:  # noqa: BLE001
        logger("FATAL: процесс упал с исключением:\n" + traceback.format_exc())
        raise
    finally:
        log_fh.close()


if __name__ == '__main__':
    main()
