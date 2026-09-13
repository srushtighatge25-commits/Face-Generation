import io
import json
import torch
from PIL import Image, ImageDraw
from app import generator, DEVICE, tensor_to_png

canvas = Image.new('RGB', (640, 736), '#f3f0e8')
draw = ImageDraw.Draw(canvas)
with torch.inference_mode():
    for seed in range(16):
        rng = torch.Generator(device=DEVICE).manual_seed(seed)
        output = generator(torch.randn(1, generator.latent_size, 1, 1, device=DEVICE, generator=rng))
        portrait = Image.open(tensor_to_png(output)).resize((160,160))
        x, y = (seed % 4) * 160, (seed // 4) * 184
        canvas.paste(portrait, (x,y))
        draw.text((x+8,y+164), f'Seed {seed}', fill='black')
canvas.save('model_review/baseline.png')
print(json.dumps({'torch': torch.__version__, 'device': str(DEVICE), 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'parameters': sum(p.numel() for p in generator.parameters()), 'latent_size': generator.latent_size}, indent=2))
