#!/usr/bin/env python3
"""Reconstruct and verify the Appendix G deterministic random-digit mappings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_SEED = 42
EQUAL_SEED = 42
FIVE_DIGIT_SEED = 137
EXPECTED_EQUAL = {
    '311': '978', '47': '72', '277': '166', '280': '304', '312': '430',
    '79': '15', '279': '119', '27': '69', '18': '70', '50': '91', '77': '29',
}
EXPECTED_FIVE = {
    '311': '85174', '47': '27084', '277': '63058', '280': '34752', '312': '54197',
    '79': '66277', '279': '52065', '27': '87228', '18': '41471', '50': '31139', '77': '48423',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Corrected ASSISTments CSV downloaded independently.')
    parser.add_argument('--output', type=Path, default=Path('reproduce-output/mappings/random-mappings.json'))
    return parser.parse_args()


def selected_skills(data: Path) -> list[str]:
    frame = pd.read_csv(data, encoding='latin1', low_memory=False, usecols=['skill_id'])
    frame = frame[frame.skill_id.notna()].copy()
    return frame.skill_id.astype(str).str.strip().value_counts().head(11).index.tolist()


def make_equal_mapping(top: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(EQUAL_SEED)
    mapping: dict[str, str] = {}
    for skill in top:
        length = len(skill)
        candidates = [f'{value:0{length}d}' for value in range(10 ** length) if f'{value:0{length}d}' != skill]
        rng.shuffle(candidates)
        mapping[skill] = next(value for value in candidates if value not in mapping.values())
    return mapping


def make_five_digit_mapping(top: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(FIVE_DIGIT_SEED)
    values = rng.choice(np.arange(10000, 100000), size=len(top), replace=False).astype(str).tolist()
    return dict(zip(top, values))


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    top = selected_skills(args.data)
    equal_mapping = make_equal_mapping(top)
    five_mapping = make_five_digit_mapping(top)
    matches_appendix_g = top == list(EXPECTED_EQUAL) and equal_mapping == EXPECTED_EQUAL and five_mapping == EXPECTED_FIVE
    payload = {
        'selection': {'sample_seed_for_paper_partition': SAMPLE_SEED, 'top_skills_in_frequency_order': top},
        'equal_character_mapping': {
            'seed': EQUAL_SEED,
            'algorithm': 'for each top skill in frequency order, enumerate all same-width decimal strings except the original, shuffle with PCG64, and take the first string not already assigned',
            'mapping': equal_mapping,
        },
        'five_digit_mapping': {
            'seed': FIVE_DIGIT_SEED,
            'algorithm': 'draw 11 unique integers without replacement from numpy.arange(10000, 100000) with PCG64, convert each to decimal text, and zip to top skills in frequency order',
            'mapping': five_mapping,
        },
        'appendix_g_exact_match': matches_appendix_g,
        'note': 'A mismatch indicates an input-file version or top-skill ordering difference and should be investigated before comparing model outputs.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
