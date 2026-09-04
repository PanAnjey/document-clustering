"""Build one reproducible, heterogeneous text benchmark sample."""

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import fitz
import pandas as pd
import psycopg2
import pytesseract
from PIL import Image
from PIL import ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from em_config import DB_URL

pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_PATH
Image.MAX_IMAGE_PIXELS = 178_956_970
ImageFile.LOAD_TRUNCATED_IMAGES = True


GROUPS = {
    'pdf_text': lambda f: f == 'pdf_text',
    'pdf_tables': lambda f: (f or '').startswith('pdf_tables'),
    'word': lambda f: (f or '').startswith('word_'),
    'excel': lambda f: (f or '').startswith('excel_'),
    'xml': lambda f: (f or '').startswith('xml_'),
}

SUPPORTED_SUFFIXES = {
    'pdf_text': {'.pdf'},
    'pdf_tables': {'.pdf'},
    'word': {'.docx', '.rtf', '.odt', '.txt'},
    'excel': {'.xlsx', '.xlsm', '.ods', '.csv'},
    'xml': {'.xml', '.xsd', '.xsl', '.xslt', '.wsdl'},
}


def clean(text: str, limit: int = 20000) -> str:
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text[:limit]


def zip_text(path: Path) -> str:
    chunks = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(('.xml', '.rels', '.txt')):
                raw = archive.read(name).decode('utf-8', errors='ignore')
                chunks.append(re.sub(r'<[^>]+>', ' ', raw))
    return clean(' '.join(chunks))


def extract(path: Path, group: str) -> str:
    if group in {'pdf_text', 'pdf_tables'}:
        with fitz.open(path) as doc:
            return clean(doc[0].get_text('text') if doc.page_count else '')
    if group == 'pdf_scan':
        with fitz.open(path) as doc:
            if not doc.page_count:
                return ''
            pix = doc[0].get_pixmap(dpi=200, alpha=False)
            image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
        return clean(pytesseract.image_to_string(image, lang='rus+eng'))
    if group == 'image':
        with Image.open(path) as image:
            return clean(pytesseract.image_to_string(image, lang='rus+eng'))
    if path.suffix.lower() in {'.docx', '.xlsx', '.xlsm', '.odt', '.ods'}:
        return zip_text(path)
    if path.suffix.lower() in {'.xml', '.xsd', '.xsl', '.xslt', '.wsdl', '.txt', '.csv'}:
        return clean(path.read_text(encoding='utf-8', errors='ignore'))
    if path.suffix.lower() == '.rtf':
        raw = path.read_text(encoding='cp1251', errors='ignore')
        return clean(re.sub(r'\\[a-z]+\d* ?', ' ', re.sub(r'[{}]', ' ', raw)))
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-group', type=int, default=250)
    ap.add_argument('--seed', type=int, default=20260818)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('mixed_sample_2000.csv'))
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    df = pd.read_sql('SELECT id, format_type, file_path FROM documents', conn)
    conn.close()
    df['source_group'] = df['format_type'].apply(
        lambda f: next((name for name, predicate in GROUPS.items() if predicate(f)), 'other')
    )
    selected = []
    for group in GROUPS:
        part = df[
            (df['source_group'] == group)
            & df['file_path'].map(lambda p: Path(p).suffix.lower() in SUPPORTED_SUFFIXES[group])
        ]
        selected.append(part.sample(n=min(args.per_group, len(part)), random_state=args.seed))
    sample = pd.concat(selected, ignore_index=True).sample(frac=1, random_state=args.seed)

    rows = []
    for number, row in enumerate(sample.itertuples(index=False), 1):
        try:
            text = extract(Path(row.file_path), row.source_group)
        except Exception as exc:  # noqa: BLE001
            print(f'{number}/{len(sample)} failed {row.id}: {type(exc).__name__}: {exc}', flush=True)
            text = ''
        if len(text) >= 30:
            rows.append({
                'id': int(row.id),
                'source_format': row.format_type,
                'source_group': row.source_group,
                'file_path': row.file_path,
                'text': text,
            })
        if number % 100 == 0:
            print(f'{number}/{len(sample)} extracted, kept {len(rows)}', flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False, encoding='utf-8')
    print(f'output={args.output} n={len(result)}', flush=True)
    print(result.groupby('source_group').size().to_string(), flush=True)
    print(f"avg_chars={result['text'].str.len().mean():.1f}", flush=True)


if __name__ == '__main__':
    main()
