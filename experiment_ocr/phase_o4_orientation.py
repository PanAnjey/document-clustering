# phase_o4_orientation.py
# Фаза O4: парный тест классификаторов ориентации на тех же 20 документах,
# что и тест устойчивости OCR (0°/90°/180°/270°).
#
# Классификаторы:
#   vit_large — ViT-Large 384, обучена в проекте (99.63% на TRAIN-сете)
#   effnet    — EfficientNet-B1, обучена в проекте (99.57%)
#   pp_lcnet  — PP-LCNet_x1_0_doc_ori (PaddleOCR, встроенная)
#
# Запуск (venv_ocr): python phase_o4_orientation.py --cls vit_large|effnet|pp_lcnet
# Итог: results/orientation_cls.jsonl (resume) + сводка в stdout.

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from oc_config import RENDER_DIR, RESULTS_DIR  # noqa: E402

N_DOCS = 20
ANGLES = [0, 90, 180, 270]
ROT_DIR = RENDER_DIR / 'rot'
OUT_JSONL = RESULTS_DIR / 'orientation_cls.jsonl'

VIT_PATH = Path(r"D:\FileOrganizer\TRAIN\models\orientation_vit_large_384_best.pth")
EFFNET_PATH = Path(r"D:\FileOrganizer\TRAIN\models_efficientnet\orientation_efficientnet_b1_best.pth")


def load_gt_docs():
    """Те же 20 документов, что в phase_o3 (GT ≥ 300 символов)."""
    gt = []
    for line in io.open(RESULTS_DIR / 'o1_results.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r['source'] == 'pdf_text':
            gt.append(r['doc_id'])
        if len(gt) >= N_DOCS:
            break
    return gt


def get_images(doc_id):
    """doc_id → {угол: путь png} (0/90/180/270; rot270 создаётся)."""
    base = RENDER_DIR / f'pdf_text_{doc_id}.png'
    ROT_DIR.mkdir(exist_ok=True)
    outs = {0: str(base)}
    img = Image.open(base)
    for ang in (90, 180, 270):
        out = ROT_DIR / f'pdf_text_{doc_id}_rot{ang}.png'
        if not out.exists():
            img.rotate(-ang, expand=True).save(out)
        outs[ang] = str(out)
    return outs


# ── Классификаторы ───────────────────────────────────────────────

class VitClassifier:
    name = 'vit_large'

    def __init__(self):
        import torch
        from torchvision import transforms
        self.device = torch.device('cuda:0')
        self.model = torch.load(VIT_PATH, map_location=self.device,
                                weights_only=False)
        self.model.to(self.device).eval()
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox(im, 384)),
            transforms.ToTensor(), norm])
        self.torch = torch

    def predict(self, png_path):
        img = Image.open(png_path).convert('RGB')
        t = self.tf(img).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            prob = self.torch.softmax(self.model(t).float(), dim=1)
            conf, pred = prob.max(dim=1)
        # Чекпоинт из retrain-линейки: классы в лексикографическом порядке
        # ['0','180','270','90'] (строковая сортировка), а не [0,90,180,270].
        # Проверено на distrib (0/90/180/270) и синтетических поворотах.
        angle = [0, 180, 270, 90][int(pred[0].item())]
        return angle, float(conf[0].item())


class EffnetClassifier:
    name = 'effnet'

    def __init__(self):
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        self.torch = torch
        self.device = torch.device('cuda:0')
        ckpt = torch.load(EFFNET_PATH, map_location=self.device,
                          weights_only=False)
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2
        model = models.efficientnet_b1(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, ckpt['num_classes'])
        model.load_state_dict(ckpt['model_state'], strict=False)
        self.model = model.to(self.device).eval()
        idx_to_class = ckpt.get('idx_to_class')
        self.idx_to_angle = ({int(k): int(v) for k, v in idx_to_class.items()}
                             if idx_to_class else {0: 0, 1: 90, 2: 180, 3: 270})
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.tf = transforms.Compose([
            transforms.Lambda(lambda im: letterbox(im, 240)),
            transforms.ToTensor(), norm])

    def predict(self, png_path):
        img = Image.open(png_path).convert('RGB')
        t = self.tf(img).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            prob = self.torch.softmax(self.model(t).float(), dim=1)
            conf, pred = prob.max(dim=1)
        return self.idx_to_angle[int(pred[0].item())], float(conf[0].item())


class PpLcnetClassifier:
    name = 'pp_lcnet'

    def __init__(self):
        from paddlex import create_model
        self.model = create_model(model_name='PP-LCNet_x1_0_doc_ori')

    def predict(self, png_path):
        res = list(self.model.predict(input=png_path))[0]
        # paddlex: dict-подобный результат с label_names / scores
        try:
            label = res['label_names'][0]
            score = float(res['scores'][0])
        except (TypeError, KeyError):
            label = str(res['pred'][0]) if 'pred' in res else str(res)
            score = 0.0
        return int(label), score


def letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (size, size), (255, 255, 255))
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


CLS_REGISTRY = {
    'vit_large': VitClassifier,
    'effnet': EffnetClassifier,
    'pp_lcnet': PpLcnetClassifier,
}


def load_done():
    done = set()
    if OUT_JSONL.exists():
        for line in io.open(OUT_JSONL, encoding='utf-8'):
            try:
                r = json.loads(line)
                done.add((r['doc_id'], r['angle'], r['cls']))
            except Exception:  # noqa: BLE001
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cls', required=True, choices=list(CLS_REGISTRY))
    args = ap.parse_args()

    doc_ids = load_gt_docs()
    print(f"документов: {len(doc_ids)}, углы: {ANGLES}, классификатор: {args.cls}")
    clf = CLS_REGISTRY[args.cls]()
    done = load_done()

    n_new = 0
    with io.open(OUT_JSONL, 'a', encoding='utf-8') as out:
        for doc_id in doc_ids:
            for true_ang, png in get_images(doc_id).items():
                if (doc_id, true_ang, args.cls) in done:
                    continue
                t0 = time.time()
                try:
                    pred, conf = clf.predict(png)
                    rec = {'doc_id': doc_id, 'angle': true_ang,
                           'cls': args.cls, 'pred': pred, 'conf': round(conf, 4),
                           'sec': round(time.time() - t0, 2)}
                except Exception as e:  # noqa: BLE001
                    rec = {'doc_id': doc_id, 'angle': true_ang,
                           'cls': args.cls, 'pred': -1, 'conf': 0.0,
                           'sec': round(time.time() - t0, 2),
                           'error': f"{type(e).__name__}: {e}"}
                out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                out.flush()
                n_new += 1
                ok = '✓' if rec['pred'] == true_ang else '✗'
                print(f"  {doc_id} rot{true_ang}: pred {rec['pred']} {ok} "
                      f"(conf {rec['conf']})", flush=True)
    print(f"done. новых: {n_new} → {OUT_JSONL}")


if __name__ == '__main__':
    main()
