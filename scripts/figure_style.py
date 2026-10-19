"""Shared, explicit typography for every manuscript figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
STYLE = json.loads((ROOT/'configs/figure_style.json').read_text(encoding='utf-8'))
FONT_FAMILY = STYLE['font_family']


def font_path() -> Path:
    """Require the requested font instead of silently changing its appearance."""
    try:
        return Path(font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),
                                         fallback_to_default=False))
    except ValueError as error:
        raise RuntimeError(f'Install {FONT_FAMILY} before rendering manuscript figures; '
                           'numerical DATA reproduction does not require this font.') from error


def apply_font() -> dict:
    path = font_path()
    font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY, weight='bold'),
                          fallback_to_default=False)
    mpl.rcParams.update({'font.family': FONT_FAMILY, 'font.sans-serif': [FONT_FAMILY],
        'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY,
        'mathtext.it': FONT_FAMILY + ':italic', 'mathtext.bf': FONT_FAMILY + ':bold',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    return {'font_family': FONT_FAMILY, 'font_file_basename': path.name,
            'silent_font_fallback_allowed': False}
