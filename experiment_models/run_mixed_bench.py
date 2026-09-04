"""Run every embedding model on the same fixed sample."""

import argparse
import subprocess
import sys
from pathlib import Path

from em_config import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', required=True)
    ap.add_argument('--sample', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-length', type=int, default=512)
    args = ap.parse_args()
    python = sys.executable
    script = Path(__file__).with_name('phase_m3_embed.py')
    for model in MODELS:
        print(f'=== {model} on {args.device} ===', flush=True)
        subprocess.run([
            python, '-u', '-X', 'utf8', str(script),
            '--model', model,
            '--device', args.device,
            '--tag', args.tag,
            '--sample', str(args.sample),
            '--max-length', str(args.max_length),
        ], check=True)


if __name__ == '__main__':
    main()
