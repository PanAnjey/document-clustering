# _run_m3_qwen.py — запуск M3 для одной модели в 2 шарда (cuda:0 + cuda:1).
# Ждёт завершения обоих. Запуск: python _run_m3_qwen.py qwen3e_4b
import subprocess
import sys

key = sys.argv[1]
flags = 0x00000008 | 0x00000200 | 0x01000000 | 0x08000000
cwd = r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\experiment_models"
py = r"D:\VENV\LLM\Scripts\python.exe"

procs = []
for shard, dev in ((0, "cuda:0"), (1, "cuda:1")):
    log = open(cwd + rf"\logs\m3_run_{key}_shard{shard}.console.log", "wb")
    p = subprocess.Popen(
        [py, "-u", "-X", "utf8", "phase_m3_embed.py",
         "--model", key, "--device", dev, "--shards", "2", "--shard-id", str(shard)],
        cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
        creationflags=flags, close_fds=True)
    print(f"shard {shard} on {dev}: PID {p.pid}", flush=True)
    procs.append((shard, p))

rc = {}
for shard, p in procs:
    rc[shard] = p.wait()
print(f"exit codes: {rc}", flush=True)
