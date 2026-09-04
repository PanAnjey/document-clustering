# oc_metrics.py
# Метрики качества OCR против эталона: CER, WER, покрытие по длине,
# + порядко-независимые (bag-of-words F1, char n-gram cosine).
#
# CER/WER чувствительны к порядку текста: у таблиц fitz и OCR выдают
# строки в разном порядке → завышенные значения. Для корпуса с таблицами
# главные метрики — BoW-F1 и 3gram-cos.
# rapidfuzz.distance.Levenshtein принимает произвольные последовательности —
# для WER подаём списки слов.

from collections import Counter

from rapidfuzz.distance import Levenshtein

from oc_textnorm import normalize, words


def cer(gt: str, hyp: str) -> float:
    g, h = normalize(gt), normalize(hyp)
    if not g:
        return 0.0
    return Levenshtein.distance(g, h) / len(g)


def wer(gt: str, hyp: str) -> float:
    g, h = words(gt), words(hyp)
    if not g:
        return 0.0
    return Levenshtein.distance(g, h) / len(g)


def coverage(gt: str, hyp: str) -> float:
    """Отношение длины OCR к длине эталона (<1 — движок теряет блоки)."""
    g, h = normalize(gt), normalize(hyp)
    if not g:
        return 0.0
    return len(h) / len(g)


def bow_f1(gt: str, hyp: str) -> float:
    """Bag-of-words F1 по мультимножествам слов (порядок не важен)."""
    g, h = Counter(words(gt)), Counter(words(hyp))
    if not g and not h:
        return 1.0
    if not g or not h:
        return 0.0
    inter = sum((g & h).values())
    p = inter / sum(h.values())
    r = inter / sum(g.values())
    return 2 * p * r / (p + r) if p + r else 0.0


def ngram_cosine(gt: str, hyp: str, n: int = 3) -> float:
    """Косинус по мультимножествам символьных n-грамм (порядок толерантен)."""
    def ngrams(t):
        t = normalize(t)
        return Counter(t[i:i + n] for i in range(max(len(t) - n + 1, 1)))
    g, h = ngrams(gt), ngrams(hyp)
    inter = sum((g & h).values())
    return inter / max((sum(g.values()) * sum(h.values())) ** 0.5, 1e-9)
