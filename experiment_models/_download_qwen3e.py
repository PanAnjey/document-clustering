# _download_qwen3e.py — скачивание только Qwen3-Embedding-4B и 0.6B.
import os
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download  # noqa: E402

from em_config import MODELS, model_local_path  # noqa: E402

for key in ('qwen3e_4b', 'qwen3e_06b'):
    local = model_local_path(key)
    print(f"=== {key}: {MODELS[key]['hf_name']} -> {local}", flush=True)
    snapshot_download(
        MODELS[key]['hf_name'],
        local_dir=str(local),
        ignore_patterns=['*.bin', 'onnx/*', 'openvino/*', 'tf_model*', 'rust_model*'],
    )
    print(f"  done: {key}", flush=True)
print("all downloaded", flush=True)
