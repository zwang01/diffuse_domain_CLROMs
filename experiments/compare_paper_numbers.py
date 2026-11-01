"""Compare results/paper_numbers.json with reference/published_numbers.json.

Usage:  python experiments/compare_paper_numbers.py [NEW_JSON] [REFERENCE_JSON]
Prints every quantity that differs from the published value by more than a
relative 1e-6. Wall times and speedups are listed separately: they depend on
the workstation.
"""
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
new_file = Path(sys.argv[1]) if len(sys.argv) > 1 else PKG / 'results' / 'paper_numbers.json'
ref_file = Path(sys.argv[2]) if len(sys.argv) > 2 else PKG / 'reference' / 'published_numbers.json'
if not new_file.exists():
    sys.exit(f'{new_file} not found: run export_paper_numbers in MATLAB first.')
ref, new = json.loads(ref_file.read_text()), json.loads(new_file.read_text())


def leaves(a, b, path=''):
    if isinstance(a, dict):
        for k in a:
            yield from leaves(a[k], b.get(k) if isinstance(b, dict) else None,
                              f'{path}.{k}' if path else k)
    elif isinstance(a, list):
        for i, x in enumerate(a):
            yield from leaves(x, b[i] if isinstance(b, list) and i < len(b) else None, f'{path}[{i}]')
    else:
        yield path, a, b


same, timing, differ = 0, [], []
for path, a, b in leaves(ref, new):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        rel = abs(a - b) / max(abs(a), 1e-300)
    else:
        rel = 0.0 if a == b else float('inf')
    if 'seconds' in path or 'speedup' in path:
        timing.append((path, a, b))
    elif rel <= 1e-6:
        same += 1
    else:
        differ.append((path, a, b, rel))

print(f'{same} quantities agree with the published values to 1e-6.')
if differ:
    print(f'\n{len(differ)} quantities differ:')
    for path, a, b, rel in differ:
        print(f'  {path}: published {a:.6g}, now {b:.6g} (relative {rel:.1e})')
print('\nTimings (machine dependent):')
for path, a, b in timing:
    print(f'  {path}: published {a:.4g}, now {b:.4g}')
