"""Check published file integrity and Python syntax; no model imports."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'manifests/published_files.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        p=ROOT/row['path']
        assert p.is_file(),row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    print('PUBLISHED_FILES_CHECK_OK',len(manifest['files']))
if __name__=='__main__':main()
