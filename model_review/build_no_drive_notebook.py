"""Create the Colab variant that uses temporary storage and manual backups."""
import json
from pathlib import Path

notebook = json.loads(Path('Fine_tune_CelebA.ipynb').read_text(encoding='utf-8'))
for cell in notebook['cells']:
    source = ''.join(cell['source'])
    if source.startswith('# Fine-tune your existing'):
        source += '\n**No Google Drive connection is required.** Outputs are stored temporarily in Colab. Download your backup after each training session; files are lost when the runtime is deleted or expires.\n'
    elif source.startswith('## Save training progress'):
        source = '''## Use temporary Colab storage
No Drive sign-in or mounting is needed. Training files are saved under `/content/face_training`.
Start with one epoch, download the backup, then increase the total epoch target to continue.
Downloaded backups are the only persistent copy if Colab deletes this runtime.
'''
    elif "drive.mount('/content/drive')" in source:
        source = '''from google.colab import files
from pathlib import Path
import os, subprocess, sys

RUN_DIR = Path('/content/face_training/celeba_finetune_01')
RUN_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = Path('/content/face_finetune')
WORK_DIR.mkdir(exist_ok=True)
os.chdir(WORK_DIR)
print('Temporary training folder:', RUN_DIR)
'''
    elif source.startswith('## Upload your original'):
        source = source.replace('On subsequent sessions the saved original is reused.', 'Within this runtime the saved original is reused. In a new runtime, restore your downloaded backup using the optional cell below, or upload the original again to begin a fresh run.')
    elif source.startswith('## Download and extract'):
        source = source.replace('outside the Google Drive run folder', 'outside the training output folder and backup')
    elif source.startswith('## Fine-tune'):
        source = source.replace('Start with 5 epochs', 'Start with 1 epoch')
        source += '\nRun the backup cell immediately afterward. Increase `EPOCHS` to 2, 3, etc. to continue from the saved checkpoint in this runtime.\n'
    elif source.startswith('EPOCHS = 5'):
        source = source.replace('EPOCHS = 5', 'EPOCHS = 1')
    cell['source'] = source.splitlines(keepends=True)

def cell(kind, text):
    result = {'cell_type':kind, 'metadata':{}, 'source':text.splitlines(keepends=True)}
    if kind == 'code': result.update(execution_count=None, outputs=[])
    return result

restore = [cell('markdown', '''## Optional: restore a previously downloaded training backup
Skip this on your first run. On a fresh runtime, run this before uploading the original.
Upload `celeba-training-backup.zip` downloaded from this notebook. Restore only your own backup.
The dataset and training code are not in the backup: run their setup cells again afterward.
Keep the same crop, batch size, and training settings; increase the target epoch to continue.
'''), cell('code', '''RESTORE_BACKUP = False  # Set True only when continuing in a new runtime.
if RESTORE_BACKUP:
    import zipfile
    if any(RUN_DIR.iterdir()):
        raise RuntimeError('Restore into an empty run folder to avoid overwriting another run.')
    print('Upload celeba-training-backup.zip')
    uploaded = files.upload()
    name = 'celeba-training-backup.zip'
    if name not in uploaded:
        raise ValueError('Expected ' + name)
    with zipfile.ZipFile(WORK_DIR / name) as archive:
        allowed = {'.pth', '.pt', '.png', '.json', '.jsonl'}
        for entry in archive.infolist():
            path = Path(entry.filename)
            if (entry.is_dir() or len(path.parts) != 1 or path.name != entry.filename
                    or path.suffix not in allowed or chr(92) in entry.filename):
                raise ValueError('Unexpected backup entry: ' + entry.filename)
        for entry in archive.infolist():
            with archive.open(entry) as source, (RUN_DIR / entry.filename).open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
    del uploaded
    print('Backup restored. Continue with the setup cells.')
''')]
notebook['cells'][4:4] = restore
backup = [cell('markdown', '''## Download a backup now
This archive includes the original generator, full resume checkpoint, candidate weights,
settings, and sample grids. It excludes Kaggle credentials and the CelebA dataset.
Wait for your browser download to finish before closing or deleting the runtime.
Rerun this cell after each additional epoch. A candidate `.pth` alone cannot resume
both optimizers and the discriminator; keep this full backup for continued training.
'''), cell('code', '''import zipfile
if not (RUN_DIR / 'latest.pt').exists():
    raise RuntimeError('No completed epoch to back up yet.')
backup = WORK_DIR / 'celeba-training-backup.zip'
with zipfile.ZipFile(backup, 'w', compression=zipfile.ZIP_STORED) as archive:
    for path in sorted(RUN_DIR.iterdir()):
        if path.is_file() and path.suffix in {'.pth', '.pt', '.png', '.json', '.jsonl'}:
            archive.write(path, arcname=path.name)
print(f'Backup: {backup.stat().st_size / 1024**2:.1f} MB')
files.download(str(backup))
''')]
for index, entry in enumerate(notebook['cells']):
    if ''.join(entry['source']).startswith('EPOCHS = 1'):
        notebook['cells'][index+1:index+1] = backup
        break
notebook['metadata']['colab']['name'] = 'Fine_tune_CelebA_No_Drive.ipynb'
for index, entry in enumerate(notebook['cells']): entry['id'] = f'cell-{index:02}'
Path('Fine_tune_CelebA_No_Drive.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
import ast
for entry in notebook['cells']:
    if entry['cell_type'] == 'code':
        source = ''.join(entry['source'])
        ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
        assert 'drive.mount(' not in source
print('Created and syntax-checked no-Drive notebook:', len(notebook['cells']), 'cells')
