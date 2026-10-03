import json
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1];m=json.loads((root/'config/model.json').read_text())
print(snapshot_download(m['repository'],revision=m['revision'],local_dir=root/'models/Qwen3-1.7B-4bit',max_workers=4))
