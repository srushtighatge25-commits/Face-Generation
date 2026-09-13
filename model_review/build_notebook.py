import json
from pathlib import Path

cells=[]
def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
def code(s): cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(keepends=True)})
md('''# Fine-tune your existing face generator on CelebA

This notebook starts from **your `generator.pth`**, not a replacement pretrained model.
It trains a new discriminator and fine-tunes the original generator at 64×64 resolution.
The output is a **candidate**: improved realism is not guaranteed until you train and compare.

Run cells in order. Select **Runtime → Change runtime type → GPU** first.
Keep your original checkpoint. No Kaggle credentials are embedded in this notebook.
''')
code('''import torch
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before downloading data or training.")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
%pip install -q kaggle Pillow
''')
md('''## Save training progress to Google Drive
Use a new run folder for each experiment. Epoch snapshots survive runtime disconnections.
A disconnect during an epoch resumes from the last completed epoch.
''')
code('''from google.colab import drive, files
from pathlib import Path
import os, subprocess, sys

drive.mount('/content/drive')
RUN_DIR = Path('/content/drive/MyDrive/Face_Generation/celeba_finetune_01')
RUN_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path('/content/face_finetune')
WORK_DIR.mkdir(exist_ok=True)
os.chdir(WORK_DIR)
''')
md('''## Upload your original generator
Upload the `generator.pth` currently used in your app (about 14 MB).
This file is copied into the run directory once and is never overwritten by training.
On subsequent sessions the saved original is reused.
''')
code('''import shutil
ORIGINAL = RUN_DIR / 'original_generator.pth'
if not ORIGINAL.exists():
    print('Upload generator.pth')
    uploaded = files.upload()
    if 'generator.pth' not in uploaded:
        raise ValueError('Expected generator.pth')
    shutil.copyfile(WORK_DIR / 'generator.pth', ORIGINAL)
    del uploaded
print('Original checkpoint:', ORIGINAL)
''')
md('''## Download and extract your CelebA dataset
Use your existing Kaggle `kaggle.json` credentials. This corrects the escaped paths
in your snippet and extracts the downloaded archive. Expect a multi-GB download.
Credentials stay in this temporary Colab runtime, outside the Google Drive run folder.
''')
code('''credential = Path.home() / '.kaggle' / 'kaggle.json'
if not credential.exists():
    print('Upload kaggle.json')
    uploaded = files.upload()
    if 'kaggle.json' not in uploaded:
        raise ValueError('Expected kaggle.json')
    credential.parent.mkdir(parents=True, exist_ok=True)
    credential.write_bytes(uploaded['kaggle.json'])
    credential.chmod(0o600)
    (WORK_DIR / 'kaggle.json').unlink(missing_ok=True)
    del uploaded
DATA_DIR = Path('/content/celeba_data')
if not (DATA_DIR / '.download_complete').exists():
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'jessicali9530/celeba-dataset',
                    '-p', str(DATA_DIR), '--unzip'], check=True)
    (DATA_DIR / '.download_complete').touch()
print('Dataset ready:', DATA_DIR)
''')
md('''## Install the checkpoint-compatible training code
The source is included below so this notebook does not need GitHub or other project files.
Only your original generator and the CelebA dataset are required.
''')
code("(WORK_DIR / 'models.py').write_text("+repr(Path('models.py').read_text(encoding='utf-8-sig'))+", encoding='utf-8')\n"+"(WORK_DIR / 'train.py').write_text("+repr(Path('train.py').read_text(encoding='utf-8-sig'))+", encoding='utf-8')\nprint('Training code ready')\n")
md('''## Check the real-image crop before training
The default follows PyTorch's DCGAN example: resize shortest side to 64,
center-crop to 64×64, normalize to [-1,1], and randomly mirror horizontally.
Your download snippet does not show your original transforms. If you previously
center-cropped to 108×108 **before resizing**, set `CROP_SIZE = 108` here.
Use the same setting when resuming. Other original transforms may require editing `FaceDataset`.
''')
code('''from train import FaceDataset
from PIL import Image
from IPython.display import display
CROP_SIZE = 0
faces = FaceDataset(DATA_DIR, crop_size=CROP_SIZE)
print('Images:', len(faces))
preview = Image.new('RGB', (8*64, 2*64))
for i in range(16):
    pixels = faces[i].add(1).mul(127.5).byte().permute(1,2,0).contiguous()
    preview.paste(Image.frombytes('RGB', (64,64), bytes(pixels.flatten().tolist())), ((i%8)*64,(i//8)*64))
display(preview.resize((768,192)))
''')
md('''## Fine-tune
Defaults are experimental starting points: 200 discriminator-only warm-up steps,
generator learning rate 0.00002, discriminator learning rate 0.0001,
spectral normalization, logistic loss, lazy R1 regularization, and an exponential
moving average (EMA) of generator weights. The original generator is not reinitialized.

Start with 5 epochs and inspect progress. Increase the total `EPOCHS` to continue.
Loss values are training diagnostics, not realism scores. Reduce `BATCH_SIZE` to 32
if you run out of GPU memory; use a fresh run folder when changing saved settings.
Each epoch saves a PNG, a 14-MB generator, and a rolling full resume checkpoint.
''')
code('''EPOCHS = 5
BATCH_SIZE = 64
command = [sys.executable, '-u', 'train.py', '--data', str(DATA_DIR),
           '--checkpoint', str(ORIGINAL), '--output', str(RUN_DIR),
           '--epochs', str(EPOCHS), '--batch-size', str(BATCH_SIZE),
           '--crop-size', str(CROP_SIZE), '--device', 'cuda']
if (RUN_DIR / 'latest.pt').exists():
    command += ['--resume', str(RUN_DIR / 'latest.pt')]
subprocess.run(command, check=True)
''')
md('''## Compare identical latent inputs before and after training
Each grid uses the same 32 latent inputs. Look for more consistent eyes, mouths,
face shapes, and backgrounds **without losing variety**. Also inspect fresh seeds
below; do not judge improvement from one attractive face. Stop if faces collapse
into near-duplicates or become less coherent. Retain the original when results regress.
''')
code('''print('BEFORE')
display(Image.open(RUN_DIR / 'before.png').resize((768,384)))
for path in sorted(RUN_DIR.glob('epoch-*.png')):
    print(path.name)
    display(Image.open(path).resize((768,384)))
''')
md('''## Evaluate a chosen epoch on fresh seeds and download
Select an epoch after comparing the grids. This does not automatically declare any
checkpoint best. For a formal quality claim, additionally compare original and
candidate on held-out real images with a consistent FID/KID protocol; this notebook
does not compute those metrics. Results remain 64×64, with the same architecture.
''')
code('''from models import load_generator
from train import save_grid
CHOSEN_EPOCH = EPOCHS  # Change after visually comparing snapshots.
chosen = RUN_DIR / f'generator-epoch-{CHOSEN_EPOCH:03d}.pth'
noise = torch.randn(32, 100, 1, 1, generator=torch.Generator().manual_seed(73129))
for label, path in [('original', ORIGINAL), ('candidate', chosen)]:
    model = load_generator(path, 'cuda')
    if model.latent_size != noise.shape[1]:
        noise = torch.randn(32, model.latent_size, 1, 1, generator=torch.Generator().manual_seed(73129))
    save_grid(model, noise.to('cuda'), WORK_DIR / f'{label}-fresh.png')
    print(label)
    display(Image.open(WORK_DIR / f'{label}-fresh.png').resize((768,384)))
files.download(str(chosen))
''')
md('''## Try the candidate in the local app
Keep the original `generator.pth`. Put the downloaded candidate in the project folder,
then run this in PowerShell (change the filename to your selected epoch):

```powershell
$env:FACE_MODEL_PATH = "generator-epoch-005.pth"
.\\venv\\Scripts\\python.exe app.py
```

To return to the original, remove the setting and restart:

```powershell
Remove-Item Env:FACE_MODEL_PATH
```

References: [PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html),
[Kaggle dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
''')
notebook={'cells':cells,'metadata':{'colab':{'name':'Fine_tune_CelebA.ipynb'},'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
for i,c in enumerate(cells): c['id']=f'cell-{i:02}'
Path('Fine_tune_CelebA.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print('Notebook created:',len(cells),'cells')
