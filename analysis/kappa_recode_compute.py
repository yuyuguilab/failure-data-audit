#!/usr/bin/env python3
"""Recompute raw agreement and Cohen's kappa for the pre-registered blind recoding (v1).

Usage (from the repository root):
    python3 analysis/kappa_recode_compute.py

Reads: analysis/output/audit_units.csv, analysis/kappa_recoding_v1_K1.json,
       analysis/kappa_recoding_v1_K2.json
Writes: analysis/kappa_recoding_v1.json (the "result" and "headline_robustness" blocks;
        "design" and "sample" are carried over from the existing file).

Comparison rules:
  - Categories are normalised (lower-case; hyphen/space -> underscore) before comparison,
    so that a spelling difference such as "not-released" vs "not_released" is not a disagreement.
  - The flagship unit code of the second coder is derived from its mechanism-level records:
    mechanisms coded "undetermined" are excluded, the unit code is the highest remaining level.
"""
import csv, json, os
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, 'output', 'audit_units.csv')
K1 = os.path.join(HERE, 'kappa_recoding_v1_K1.json')
K2 = os.path.join(HERE, 'kappa_recoding_v1_K2.json')
OUT = os.path.join(HERE, 'kappa_recoding_v1.json')
UID = '\ufeffunit_id'

K1_IDS = OrderedDict([('U1', 'A-01'), ('U2', 'A-05'), ('U3', 'A-12'), ('U4', 'A-16'),
                      ('U5', 'A-23'), ('U6', 'A-19'), ('U7', 'A-18'), ('U8', 'A-24'), ('U9', 'A-25')])
FIELDS = ['level_main', 'multi_level', 'granularity', 'attribution', 'reflection',
          'domain', 'openness', 'vehicle']
ORDER = ['l0', 'l1', 'l2', 'l3', 'l4']


def norm(v):
    return str(v or '').strip().lower().replace('-', '_').replace(' ', '_')


def kappa(pairs):
    """Cohen's kappa. Returns (raw agreement, kappa, n)."""
    n = len(pairs)
    if n == 0:
        return float('nan'), float('nan'), 0
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1.0:            # both coders used a single category throughout
        return po, 1.0, n
    return po, (po - pe) / (1 - pe), n


def main():
    orig = {r[UID]: r for r in csv.DictReader(open(UNITS, encoding='utf-8'))}
    k1 = {d['unit_id']: d for d in json.load(open(K1, encoding='utf-8'))}
    k2 = json.load(open(K2, encoding='utf-8'))

    pairs = {f: [] for f in FIELDS}
    disagreements = []
    for local, gid in K1_IDS.items():
        o, nw = orig[gid], k1[local]
        for f in FIELDS:
            a, b = norm(o.get(f)), norm(nw.get(f))
            pairs[f].append((a, b))
            if a != b:
                disagreements.append([gid, o['unit_name'], f, a, b])

    k2_levels = sorted({norm(r['level_main']) for r in k2 if norm(r['level_main']) in ORDER}, key=ORDER.index)
    b = orig['B-01']
    for f, val in [('level_main', k2_levels[-1]), ('multi_level', '+'.join(k2_levels))]:
        a = norm(b.get(f))
        pairs[f].append((a, val))
        if a != val:
            disagreements.append(['B-01', b['unit_name'], f, a, val])

    overall = [p for f in FIELDS for p in pairs[f]]
    result = {
        'per_field': {f: dict(zip(('po', 'kappa', 'n'), (lambda t: (round(t[0], 3), round(t[1], 3), t[2]))(kappa(pairs[f]))))
                      for f in FIELDS},
        'overall': dict(zip(('po', 'kappa', 'n'), (lambda t: (round(t[0], 3), round(t[1], 3), t[2]))(kappa(overall)))),
        'disagreements': disagreements,
        'k2_mech_levels': k2_levels, 'k2_main': k2_levels[-1], 'k2_n_mech': len(k2),
    }
    prev = json.load(open(OUT, encoding='utf-8'))
    prev['result'] = result
    json.dump(prev, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print('field             n   raw   kappa')
    for f in FIELDS:
        t = result['per_field'][f]
        print(f'{f:16s} {t["n"]:3d}  {t["po"]:.3f}  {t["kappa"]:.3f}')
    t = result['overall']
    print(f'{"OVERALL":16s} {t["n"]:3d}  {t["po"]:.3f}  {t["kappa"]:.3f}')
    print(f'disagreements: {len(disagreements)} of {t["n"]}')


if __name__ == '__main__':
    main()
