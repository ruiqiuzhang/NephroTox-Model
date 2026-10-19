"""Package source, parameter-search designs and input DATA; exclude results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {'README.md', 'AGENTS.md', 'requirements.txt', 'requirements-figures.txt', 'requirements-modeling.txt', 'reproduce.py', '.gitignore', '.gitattributes'}
TOP_DIRECTORIES = {'.github', 'configs', 'data', 'docs', 'figures', 'scripts', 'src', 'tests'}


def release_files(root=ROOT):
    files = []
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if any(part in {'__pycache__', 'node_modules', '.venv', 'venv', '.git', 'outputs', 'source_data', 'results', 'build', 'dist'} for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink() or path.suffix in {'.pyc', '.pyo', '.tmp', '.zip', '.log', '.joblib', '.pt', '.pth', '.npz', '.png', '.jpg', '.jpeg', '.tiff', '.pdf', '.svg'}:
            continue
        if len(relative.parts) == 1 and path.name not in TOP_FILES:
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in TOP_DIRECTORIES:
            continue
        if path.stat().st_size >= 95_000_000:
            raise ValueError(f'Unexpected large release file: {relative.as_posix()}')
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'outputs/NephroTox_1673_GitHub_Source.zip')
    args = parser.parse_args()
    files = release_files()
    expected = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    actual = hashlib.sha256((ROOT / expected['source_file']).read_bytes()).hexdigest()
    if actual != expected['sha256']:
        raise ValueError('Release DATA source/integrity mismatch')
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Write the ZIP outside the upload directory')
    sums = '\n'.join(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(ROOT).as_posix()}' for file in files) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            archive.write(file, 'NephroTox_1673_GitHub/' + file.relative_to(ROOT).as_posix())
        archive.writestr('NephroTox_1673_GitHub/SHA256SUMS.txt', sums)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
    package = {'archive': args.output.name, 'files': len(files) + 1, 'bytes': args.output.stat().st_size,
               'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'source_sha256': actual,
               'no_historical_inputs': True, 'cache_or_environment_files_included': False}
    package['generated_results_included'] = False
    args.output.with_suffix('.manifest.json').write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(package, ensure_ascii=False))


if __name__ == '__main__':
    main()
