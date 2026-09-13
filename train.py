"""Fine-tune an existing 64x64 DCGAN generator on a local face dataset."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset

from models import load_generator


class FaceDataset(Dataset):
    def __init__(self, root, crop_size=0):
        self.paths = sorted(p for p in Path(root).rglob('*')
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file())
        if not self.paths:
            raise ValueError(f'No face images found under {root}. Extract the dataset first.')
        self.crop_size = crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert('RGB')
            if self.crop_size:
                # Optional original-resolution face crop, only if this matches training.
                image = ImageOps.fit(image, (self.crop_size, self.crop_size),
                                     method=Image.Resampling.BILINEAR, centering=(.5, .5)) if min(image.size) < self.crop_size else image.crop((
                    (image.width-self.crop_size)//2, (image.height-self.crop_size)//2,
                    (image.width+self.crop_size)//2, (image.height+self.crop_size)//2))
            # Resize shortest edge to 64, then center crop, as in the DCGAN tutorial.
            w, h = image.size
            image = image.resize((int(w*64/min(w,h)), int(h*64/min(w,h))), Image.Resampling.BILINEAR)
            x, y = (image.width-64)//2, (image.height-64)//2
            image = image.crop((x, y, x+64, y+64))
            if torch.rand(()).item() < .5:
                image = ImageOps.mirror(image)
            values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return values.reshape(64, 64, 3).permute(2, 0, 1).float().div(127.5).sub(1)


class Discriminator(nn.Module):
    """Fresh critic: spectral normalization, logits, no batch normalization."""
    def __init__(self, width=64):
        super().__init__()
        layers = []
        channels = 3
        for out in (width, width*2, width*4, width*8):
            layers += [spectral_norm(nn.Conv2d(channels, out, 4, 2, 1)), nn.LeakyReLU(.2)]
            channels = out
        layers += [spectral_norm(nn.Conv2d(channels, 1, 4))]
        self.main = nn.Sequential(*layers)

    def forward(self, images):
        return self.main(images).flatten()


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1-decay)
    # Use current BN statistics; averaging counters/variances with weights is unsafe.
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


@torch.no_grad()
def save_grid(model, noise, path):
    images = model(noise).clamp(-1, 1).add(1).mul(127.5).byte().cpu()
    canvas = Image.new('RGB', (8*64, ((len(images)+7)//8)*64))
    for index, tensor in enumerate(images):
        pixels = bytes(tensor.permute(1,2,0).contiguous().view(-1).tolist())
        canvas.paste(Image.frombytes('RGB', (64,64), pixels), ((index%8)*64, (index//8)*64))
    canvas.save(path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.warmup_steps < 0:
        raise ValueError('Use positive epochs, batch size >= 2, and nonnegative workers/warmup.')
    if args.r1_interval < 1 or args.r1_gamma < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError('Invalid R1 or EMA settings.')
    if min(args.lr_g, args.lr_d) <= 0 or args.crop_size < 0 or args.max_batches < 0:
        raise ValueError('Learning rates must be positive; crop size and max batches nonnegative.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'latest.pt').exists() and not args.resume:
        raise ValueError('Output already has a run. Use --resume or choose a new output folder.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Choose a GPU runtime in Colab.')
    torch.manual_seed(args.seed)
    dataset = FaceDataset(args.data, args.crop_size)
    if len(dataset) < args.batch_size:
        raise ValueError('Dataset contains fewer images than the batch size.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=device.type == 'cuda')
    model = load_generator(args.checkpoint, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    critic = Discriminator(args.critic_width).to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr_g, betas=(0., .99))
    opt_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0., .99))
    fixed = torch.randn(32, model.latent_size, 1, 1, generator=torch.Generator().manual_seed(2026)).to(device)
    start_epoch, step = 0, 0
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    config = vars(args).copy()
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        for key in ('batch_size', 'lr_g', 'lr_d', 'crop_size', 'warmup_steps', 'r1_interval',
                    'r1_gamma', 'ema_decay', 'critic_width', 'max_batches', 'seed'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume setting {key} differs from the saved run.')
        if state['source_sha256'] != checkpoint_hash:
            raise ValueError('Resume requires the same original generator checkpoint.')
        model.load_state_dict(state['generator'])
        ema.load_state_dict(state['ema'])
        critic.load_state_dict(state['discriminator'])
        opt_g.load_state_dict(state['optimizer_g'])
        opt_d.load_state_dict(state['optimizer_d'])
        start_epoch, step = state['epoch'], state['step']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if device.type == 'cuda' and state['rng_cuda']:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
    else:
        save_grid(ema, fixed, output/'before.png')
        (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'{len(dataset)} images; {device}; starting epoch {start_epoch+1}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        sums = {'d_loss': 0., 'g_loss': 0., 'r1': 0.}
        count, g_count = 0, 0
        for real in loader:
            real = real.to(device, non_blocking=True)
            batch = len(real)
            # G stays in eval for D-only updates, preserving pretrained BN statistics.
            model.eval()
            critic.train().requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake = model(torch.randn(batch, model.latent_size, 1, 1, device=device))
            regularize = step % args.r1_interval == 0 and args.r1_gamma > 0
            real.requires_grad_(regularize)
            real_logits = critic(real)
            d_loss = F.softplus(-real_logits).mean() + F.softplus(critic(fake)).mean()
            penalty = torch.zeros((), device=device)
            if regularize:
                gradient = torch.autograd.grad(real_logits.sum(), real, create_graph=True)[0]
                penalty = gradient.square().flatten(1).sum(1).mean()
                d_loss = d_loss + .5*args.r1_gamma*args.r1_interval*penalty
            if not torch.isfinite(d_loss):
                raise RuntimeError('Nonfinite discriminator loss; resume the last saved epoch at revised settings in a new run.')
            d_loss.backward()
            opt_d.step()
            g_loss = torch.zeros((), device=device)
            if step >= args.warmup_steps:
                model.train()
                # Freeze D parameters and spectral-norm state while updating G.
                critic.eval().requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                fake = model(torch.randn(batch, model.latent_size, 1, 1, device=device))
                g_loss = F.softplus(-critic(fake)).mean()
                if not torch.isfinite(g_loss):
                    raise RuntimeError('Nonfinite generator loss. Last completed epoch is preserved.')
                g_loss.backward()
                opt_g.step()
                update_ema(ema, model, args.ema_decay)
                g_count += 1
            sums['d_loss'] += d_loss.item()
            sums['g_loss'] += g_loss.item()
            sums['r1'] += penalty.item()
            count += 1
            step += 1
            if count % 100 == 0:
                print(f'Epoch {epoch+1}, batch {count}/{len(loader)}, D {d_loss.item():.3f}, G {g_loss.item():.3f}', flush=True)
            if args.max_batches and count >= args.max_batches:
                break
        row = {'epoch': epoch+1, 'step': step, 'd_loss': sums['d_loss']/count,
               'g_loss': sums['g_loss']/max(g_count,1), 'r1': sums['r1']/count, 'g_updates': g_count}
        with (output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        save_grid(ema, fixed, output/f'epoch-{epoch+1:03d}.png')
        atomic_save(ema.state_dict(), output/f'generator-epoch-{epoch+1:03d}.pth')
        atomic_save(ema.state_dict(), output/'generator_candidate.pth')
        atomic_save({'generator': model.state_dict(), 'ema': ema.state_dict(),
                     'discriminator': critic.state_dict(), 'optimizer_g': opt_g.state_dict(),
                     'optimizer_d': opt_d.state_dict(), 'epoch': epoch+1, 'step': step,
                     'rng_cpu': torch.get_rng_state(),
                     'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                     'config': config, 'source_sha256': checkpoint_hash}, output/'latest.pt')
        print(json.dumps(row), flush=True)
    print('Finished. Compare samples before choosing a candidate; GAN loss is not an image-quality score.', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='training_output')
    p.add_argument('--resume', default='')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=20, help='Total target epochs, including resumed epochs')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr-g', type=float, default=.00002)
    p.add_argument('--lr-d', type=float, default=.0001)
    p.add_argument('--warmup-steps', type=int, default=200)
    p.add_argument('--r1-gamma', type=float, default=1.)
    p.add_argument('--r1-interval', type=int, default=16)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--crop-size', type=int, default=0, help='0: standard resize/center crop; otherwise original-resolution square crop')
    p.add_argument('--critic-width', type=int, default=64)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--max-batches', type=int, default=0, help='Smoke testing only; 0 uses the full epoch')
    return p


if __name__ == '__main__':
    train(parser().parse_args())
