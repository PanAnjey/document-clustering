# -*- coding: utf-8 -*-
# prep_clean.py -- глубокая чистка меток TRAIN2 + пул + 3 групповых сплита.
#   python prep_clean.py            # с LLM-подтверждением (нужен GPU, ~2 мин)
#   python prep_clean.py --no-llm   # только ансамбль + имя файла
#
# Правило (см. common.py): >=9/13 вариантов относят документ к чужому типу.
#   + второй сигнал (ключевые слова в имени ИЛИ zero-shot LLM) на тот же тип
#     -> ПЕРЕМЕТИТЬ; иначе -> УДАЛИТЬ.

import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from common import (DECISIONS_CSV, DUP_CSV, DUP_GROUPS, ENSEMBLE_THR, GROUP_CAP,
                    LABELS, MAX_TEXT_CHARS_EVAL, MODEL_PATH, POOL_PATH, PREDS_ALL,
                    PROMPT, SEEDS, SPLIT_CSV, TEST_FRAC, TEXTS_CACHE, extract_json)

# Ключевые слова в имени файла -> тип. Имена в TRAIN2 в основном
# человекочитаемые («Акт сверки взаиморасчетов № 63 ...»). Намеренно НЕ
# включаем «проект договора» (размытая пара Contract/ContractAppendix) --
# там решает только LLM.
FN_PATTERNS = {
    'ReconciliationAct':      r'акт\s*свер|сверк\w*\s*взаимо',
    'ProformaInvoice':        r'сч[ёе]т\s*(на\s*оплату|№|на\s*предоплату)|^сч[ёе]т\b',
    'Letter':                 r'письмо|информац\w*\s*письмо|сопроводительн',
    'PowerOfAttorney':        r'доверенност',
    'Torg12':                 r'торг[\s-]*12|товарн\w*\s*накладн|^накладная',
    'AcceptanceCertificate':  r'акт\s*(сдач|при[ёе]м|выполн|оказан\w*\s*услуг)|акт\s*кс|кс-?[23]\b|^упд\b|универсальн\w*\s*передаточн',
    'CertificateRegistry':    r'реестр',
    'ServiceDetails':         r'детализац|расшифровк|справка-?\s*расч[ёе]т',
    'SupplementaryAgreement': r'дополнительн\w*\s*соглашен|доп\.?\s*соглашен',
    'ContractAppendix':       r'приложение\s*(№|\bк\b\s*договор|\bот\b)|спецификац\w*\s*(№|\bк\b)',
    'Notification':           r'уведомлен',
    'Claim':                  r'претензи',
    'ActDisagreement':        r'акт\s*разноглас|протокол\s*разноглас',
    'PriceListAgreement':     r'прайс|тарифн\w*\s*план|соглашение\s*о\s*(прайс|тариф)',
    'Contract':               r'^договор\b|договор\s*(возмездн|подряд|аренд|поставк|оказани|№)',
    'TrustConnectionRequest': r'заявк\w*.*подключ|заявление\s*на\s*подключen|доверенн\w*\s*подключ',
}
FN_RE = {k: re.compile(v, re.IGNORECASE) for k, v in FN_PATTERNS.items()}


def fn_type(basename):
    hits = [k for k, rx in FN_RE.items() if rx.search(basename)]
    return hits[0] if len(hits) == 1 else None      # только однозначное совпадение


def load_rows():
    rows = list(csv.DictReader(open(TEXTS_CACHE, encoding='utf-8')))
    preds = json.load(open(PREDS_ALL, encoding='utf-8'))
    variants = [v for v in preds if len(preds[v]) == len(rows)]
    dg = {r['path']: r for r in csv.DictReader(open(DUP_GROUPS, encoding='utf-8'))}
    keep = []
    for i, r in enumerate(rows):
        if r['cls'] not in LABELS:
            continue
        r['idx'] = i
        r['basename'] = os.path.basename(r['path'])
        r['group_id'] = int(dg[r['path']]['group_id'])
        r['ens_votes'] = {}
        keep.append(r)
    print(f'{len(keep)} документов в {len(LABELS)} классах, вариантов ансамбля {len(variants)}')
    for r in keep:
        cnt = {}
        for v in variants:
            p = preds[v][r['idx']]
            cnt[p] = cnt.get(p, 0) + 1
        top = max(cnt, key=cnt.get)
        r['ens_top'] = top
        r['ens_n'] = cnt[top]
        r['n_var'] = len(variants)
    return keep


def llm_adjudicate(flagged):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    print(f'LLM-подтверждение {len(flagged)} помеченных...', flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, device_map={'': 0}, dtype=torch.bfloat16, attn_implementation='eager')
    model.eval()
    out = {}
    for j, r in enumerate(flagged):
        prompt = PROMPT.format(text=r['text'][:MAX_TEXT_CHARS_EVAL])
        enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                      add_generation_prompt=True, enable_thinking=False,
                                      return_dict=True, return_tensors='pt')
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=30, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        txt = tok.decode(g[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        p = extract_json(txt)
        out[r['idx']] = p.get('type') if p else None
        if (j + 1) % 20 == 0:
            print(f'  {j + 1}/{len(flagged)}', flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def make_split(dg, seed):
    rng = np.random.default_rng(seed)
    split = {}
    for _typ, sub in dg.groupby('type'):
        gsz = sub.groupby('group_id').size().to_dict()
        singles = [g for g, s in gsz.items() if s == 1]
        smalls = [g for g, s in gsz.items() if 2 <= s <= GROUP_CAP]
        larges = [g for g, s in gsz.items() if s > GROUP_CAP]
        for g in larges:
            split[g] = 'train'
        rng.shuffle(singles)
        k = round(len(singles) * TEST_FRAC)
        for g in singles[:k]:
            split[g] = 'test'
        for g in singles[k:]:
            split[g] = 'train'
        rng.shuffle(smalls)
        target = sum(gsz[g] for g in smalls) * TEST_FRAC
        acc = 0
        for g in smalls:
            if acc < target:
                split[g] = 'test'
                acc += gsz[g]
            else:
                split[g] = 'train'
        allg = singles + smalls + larges
        if len(allg) >= 2:
            if all(split[g] == 'train' for g in allg):
                split[min(allg, key=lambda x: gsz[x])] = 'test'
            elif all(split[g] == 'test' for g in allg):
                split[min(allg, key=lambda x: gsz[x])] = 'train'
    dg = dg.copy()
    dg['split'] = dg['group_id'].map(split)
    assert int((dg.groupby('group_id')['split'].nunique() > 1).sum()) == 0
    dg.to_csv(str(SPLIT_CSV).format(seed=seed), index=False, encoding='utf-8')
    te = dg[dg['split'] == 'test']
    print(f'  сид {seed}: train {(dg["split"] == "train").sum()}  test {len(te)} '
          f'(независимых {(te["group_size"] == 1).sum()}, с близнецами {(te["group_size"] > 1).sum()})')


def main():
    use_llm = '--no-llm' not in sys.argv
    rows = load_rows()

    flagged = [r for r in rows if r['ens_top'] != r['cls'] and r['ens_n'] >= ENSEMBLE_THR]
    print(f'\nпомечено ансамблем (>={ENSEMBLE_THR}/13 на чужой тип): {len(flagged)}')

    llm = llm_adjudicate(flagged) if use_llm else {}

    decisions = []
    for r in flagged:
        target = r['ens_top']
        fnt = fn_type(r['basename'])
        lp = llm.get(r['idx'])
        second = (fnt == target) or (lp == target)
        decision = f'relabel:{target}' if second else 'delete'
        decisions.append({
            'idx': r['idx'], 'basename': r['basename'], 'folder': r['cls'],
            'ensemble': target, 'votes': r['ens_n'], 'filename_type': fnt or '',
            'llm': lp or '', 'decision': decision, 'group_id': r['group_id'],
        })
    dec_by_idx = {d['idx']: d for d in decisions}

    with open(DECISIONS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['folder', 'ensemble', 'votes', 'filename_type',
                                          'llm', 'decision', 'basename'])
        w.writeheader()
        for d in sorted(decisions, key=lambda x: (x['folder'], -x['votes'])):
            w.writerow({k: d[k] for k in w.fieldnames})

    n_rel = sum(1 for d in decisions if d['decision'].startswith('relabel'))
    n_del = len(decisions) - n_rel
    print(f'  переметить: {n_rel}   удалить: {n_del}')

    # ---- сборка пула ----
    pool = []
    for r in rows:
        d = dec_by_idx.get(r['idx'])
        if d is None:
            typ = r['cls']
        elif d['decision'] == 'delete':
            continue
        else:
            typ = d['decision'].split(':', 1)[1]
        pool.append({'entity_id': str(r['idx']), 'type': typ, 'text': r['text'],
                     'file_name': r['basename'], 'group_id': r['group_id']})

    gcount = {}
    for p in pool:
        gcount[p['group_id']] = gcount.get(p['group_id'], 0) + 1
    for p in pool:
        p['group_size'] = gcount[p['group_id']]

    with open(POOL_PATH, 'w', encoding='utf-8') as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    dgdf = pd.DataFrame([{'entity_id': p['entity_id'], 'type': p['type'],
                          'group_id': p['group_id'], 'group_size': p['group_size']}
                         for p in pool])
    dgdf.to_csv(DUP_CSV, index=False, encoding='utf-8')

    import collections
    print(f'\nпул после чистки: {len(pool)} документов (было {len(rows)})')
    c0 = collections.Counter(r['cls'] for r in rows)
    c1 = collections.Counter(p['type'] for p in pool)
    print('класс: было -> стало')
    for k in sorted(LABELS):
        print(f'  {k:24s} {c0[k]:4d} -> {c1[k]:4d}  ({c1[k] - c0[k]:+d})')

    print('\nгрупповые сплиты:')
    for seed in SEEDS:
        make_split(dgdf, seed)

    # ---- случайная выборка решений ----
    import random
    random.seed(0)
    sample = random.sample(decisions, min(32, len(decisions)))
    print('\nслучайная выборка решений (folder -> ensemble | fn | llm | decision):')
    for d in sorted(sample, key=lambda x: x['decision']):
        print(f'  {d["folder"]:22s} -> {d["ensemble"]:22s} v{d["votes"]:2d} | '
              f'fn={d["filename_type"] or "-":22s} | llm={d["llm"] or "-":22s} | {d["decision"]}')
    print(f'\n-> {POOL_PATH}\n-> {DECISIONS_CSV}\nдальше: run_pipeline.py')


if __name__ == '__main__':
    main()
