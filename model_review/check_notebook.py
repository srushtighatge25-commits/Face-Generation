import ast
import json
from pathlib import Path
notebook=json.loads(Path('Fine_tune_CelebA.ipynb').read_text(encoding='utf-8'))
for i,cell in enumerate(notebook['cells']):
    if cell['cell_type']=='code':
        source=''.join(cell['source'])
        source='\n'.join(line for line in source.splitlines() if not line.startswith('%'))
        ast.parse(source)
        if "write_text(" in source and 'models.py' in source:
            tree=ast.parse(source)
            for statement in tree.body:
                if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute) and statement.value.func.attr=='write_text':
                    ast.parse(statement.value.args[0].value)
print('All notebook cells and embedded training modules parse successfully.')
