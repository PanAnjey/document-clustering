# engines/oc_paddle.py
# Движок PaddleOCR (CPU, lang='ru' — кириллическая модель PP-OCR).
#
# Совместимость API: paddleocr 3.x (predict → объекты с 'rec_texts')
# и 2.x (ocr → вложенные списки [box, (text, score)]).
#
# Модели скачиваются при первом запуске в %USERPROFILE%\.paddlex
# (кэш моделей, как у HuggingFace).

# NB: paddle 3.3.x + PP-OCRv5 на CPU падал с NotImplementedError
# ConvertPirAttribute2RuntimeAttribute (баг oneDNN-бэкенда, SO 79884564).
# Решено даунгрейдом paddlepaddle==3.2.0 — MKLDNN работает (71 с/стр → ~3 с).
# Флаг-обход на случай возврата 3.3.x: PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0.

_instance = None


def _make_ocr():
    from paddleocr import PaddleOCR
    try:  # paddleocr 3.x
        return PaddleOCR(
            lang='ru',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ), 'v3'
    except TypeError:  # paddleocr 2.x
        return PaddleOCR(lang='ru', use_angle_cls=True, show_log=False), 'v2'


def _extract(result) -> str:
    texts = []
    for res in result:
        # paddleocr 3.x: dict-подобный объект с rec_texts
        try:
            rt = res['rec_texts']
            if rt:
                texts.extend(rt)
            continue
        except (TypeError, KeyError):
            pass
        # paddleocr 2.x: список строк [box, (text, score)]
        try:
            for line in res:
                if (isinstance(line, (list, tuple)) and len(line) >= 2
                        and isinstance(line[1], (list, tuple))):
                    texts.append(line[1][0])
        except TypeError:
            pass
    return '\n'.join(texts)


class PaddleEngine:
    name = 'paddle'

    def __init__(self):
        self._ocr, self._api = _make_ocr()

    def ocr(self, png_path: str) -> str:
        if self._api == 'v3':
            return _extract(self._ocr.predict(png_path)).strip()
        return _extract(self._ocr.ocr(png_path)).strip()


def get_engine():
    global _instance
    if _instance is None:
        _instance = PaddleEngine()
    return _instance
