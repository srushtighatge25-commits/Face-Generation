from pathlib import Path
p = Path('app.py')
s = p.read_text(encoding='utf-8-sig')
start = s.index('class Generator(')
end = s.index('\ngenerator = load_generator()')
block = s[start:end]
block = block.replace('def load_generator() -> Generator:', 'def load_generator(model_path: Path | str, device: torch.device | str = "cpu") -> Generator:\n    model_path = Path(model_path)')
block = block.replace('MODEL_PATH', 'model_path').replace('DEVICE', 'device')
Path('models.py').write_text('"""Shared checkpoint-compatible DCGAN model, without web-server startup."""\nfrom __future__ import annotations\n\nfrom pathlib import Path\nfrom typing import Any\n\nimport torch\nfrom torch import nn\n\n\n' + block, encoding='utf-8')
s = s[:start] + s[end:]
s = s.replace('from typing import Any\n', '').replace('import torch.nn as nn\n', '')
s = s.replace('from PIL import Image\n', 'from PIL import Image\n\nfrom models import Generator, load_generator\n')
s = s.replace('generator = load_generator()', 'generator = load_generator(MODEL_PATH, DEVICE)')
p.write_text(s, encoding='utf-8')
