# download_models.py
# Скрипт скачивания моделей nomic-embed-text-v1.5 и nomic-embed-vision-v1.5
# Запуск: python download_models.py
#
# Использует snapshot_download вместо from_pretrained,
# чтобы обойти баг all_tied_weights_keys в кастомном коде nomic.

import os
import json
import re
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODELS_BASE_DIR = Path(r"D:\MODELS\Transformers")
MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "local_dir": MODELS_BASE_DIR / "nomic-ai_nomic-embed-text-v1.5",
        "needs_trust": True,
        "needs_einops": True,
    },
    {
        "name": "nomic-ai/nomic-embed-vision-v1.5",
        "local_dir": MODELS_BASE_DIR / "nomic-ai_nomic-embed-vision-v1.5",
        "needs_trust": True,
        "needs_einops": True,
        "config_fix": True,
    },
]


def fix_ninner_in_file(config_path: Path):
    if not config_path.exists():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        if re.search(r'"n_inner"\s*:\s*2048\.0', content):
            fixed = re.sub(r'"n_inner"\s*:\s*2048\.0', '"n_inner": 2048', content)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(fixed)
            print(f"       Fixed n_inner in: {config_path}")
            return True
    except Exception as e:
        print(f"       Warning: could not fix {config_path}: {e}")
    return False


def fix_ninner_everywhere():
    hf_cache = Path.home() / ".cache" / "huggingface"
    fixed = 0
    for search_dir in [hf_cache, MODELS_BASE_DIR]:
        if not search_dir.exists():
            continue
        for config_path in search_dir.rglob("config.json"):
            if fix_ninner_in_file(config_path):
                fixed += 1
    if fixed:
        print(f"       Fixed n_inner in {fixed} config file(s).")


def download_model(model_name: str, local_dir: Path, needs_trust: bool,
                   needs_einops: bool, config_fix: bool = False):
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"[SKIP] {model_name} -> {local_dir} (already downloaded)")
        return True

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DOWN] {model_name} -> {local_dir}")

    if needs_einops:
        try:
            import einops
        except ImportError:
            print("       Installing einops...")
            os.system(f"{os.sys.executable} -m pip install einops -q")

    from huggingface_hub import snapshot_download

    print(f"       Downloading files via snapshot_download...")
    cache_path = snapshot_download(
        model_name,
        local_dir=str(local_dir),
    )

    print(f"       Files downloaded to: {cache_path}")

    fix_ninner_everywhere()

    if config_fix:
        fix_ninner_in_file(local_dir / "config.json")

    print(f"[DONE] {model_name} saved to {local_dir}")
    return True


def verify_models():
    print("\n" + "=" * 60)
    print("Verifying model files...")
    print("=" * 60)

    all_ok = True
    for m in MODELS:
        local_dir = m["local_dir"]
        name = m["name"]
        safetensors = list(local_dir.glob("*.safetensors"))
        config = local_dir / "config.json"

        if not safetensors:
            print(f"[FAIL] {name}: no .safetensors files found in {local_dir}")
            all_ok = False
            continue

        if not config.exists():
            print(f"[FAIL] {name}: config.json not found in {local_dir}")
            all_ok = False
            continue

        total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        print(f"[OK] {name}: {len(safetensors)} safetensors, config.json present, "
              f"total size: {total_size / (1024**3):.2f} GB")

    return all_ok


def main():
    fix_ninner_everywhere()

    print("=" * 60)
    print("Downloading models to:", MODELS_BASE_DIR)
    print("=" * 60)

    all_ok = True
    for m in MODELS:
        try:
            ok = download_model(
                m["name"],
                m["local_dir"],
                m.get("needs_trust", False),
                m.get("needs_einops", False),
                m.get("config_fix", False),
            )
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"[FAIL] {m['name']}: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    fix_ninner_everywhere()

    if not verify_models():
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("All models downloaded and verified successfully.")
    else:
        print("Some models failed. Check errors above.")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()