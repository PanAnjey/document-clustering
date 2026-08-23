# download_em_models.py
# Скачивание моделей сравнительного эксперимента в D:\MODELS\Transformers
# (соглашение проекта: имя каталога = hf_name с '/' → '_').
#
# Запуск: python download_em_models.py

import os
# На время скачивания offline-флаги должны быть ВЫКЛЮЧЕНЫ
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download  # noqa: E402

from em_config import DOWNLOAD_MODELS, MODELS, model_local_path  # noqa: E402


def main():
    for key in DOWNLOAD_MODELS:
        local = model_local_path(key)
        print(f"\n=== {key}: {MODELS[key]['hf_name']} → {local}")
        snapshot_download(
            MODELS[key]['hf_name'],
            local_dir=str(local),
            ignore_patterns=['*.bin', 'onnx/*', 'openvino/*', 'tf_model*', 'rust_model*'],
        )
        print(f"  done: {key}")
    print("\nall models downloaded")


if __name__ == '__main__':
    main()
