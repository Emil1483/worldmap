#!/usr/bin/env python3
"""Clean a single-path world-map SVG.

The source map (World_map_blank_without_borders.svg) is one <path> whose `d`
attribute holds ~1171 closed subpaths. Under the default nonzero fill rule:

  * land / islands  -> negative signed area (clockwise in SVG's y-down space)
  * lakes (holes)   -> positive signed area (cut out of the land they sit in)

This tool removes:
  * islands smaller than --min-island-area
  * lakes   smaller than --min-lake-area   (filling them in as solid land)
  * Antarctica: every subpath whose top edge sits below --antarctica-y

Because most subpath movetos are *relative* (`m`), dropping one would shift
every later subpath. To stay correct, each kept subpath is re-emitted with an
absolute moveto (its true start point), so removals never disturb neighbours.
"""

import argparse
import re
import sys

TOK_RE = re.compile(r'[MmLlHhVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)')
SUBPATH_SPLIT_RE = re.compile(r'(?=[Mm])')      # split keeping the moveto
LEAD_MOVE_RE = re.compile(
    r'^\s*([Mm])\s*([-+]?(?:\d*\.\d+|\d+\.?))[\s,]*([-+]?(?:\d*\.\d+|\d+\.?))')


def parse_subpaths(d):
    """Return a list of dicts: {start:(x,y), pts:[...], area, ymin, ymax}.

    Order matches the order of subpaths in `d`. Geometry is computed in
    absolute coordinates by walking the relative/absolute command stream.
    """
    toks = TOK_RE.findall(d)
    subs = []
    cur = None
    x = y = sx = sy = 0.0
    cmd = None
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in 'Zz':
                x, y = sx, sy
            continue
        if cmd in 'Mm':
            nx, ny = float(toks[i]), float(toks[i + 1]); i += 2
            if cmd == 'm': x += nx; y += ny
            else: x, y = nx, ny
            if cur is not None:
                subs.append(cur)
            cur = {'start': (x, y), 'pts': [(x, y)]}
            sx, sy = x, y
            cmd = 'l' if cmd == 'm' else 'L'   # subsequent pairs are linetos
        elif cmd in 'Ll':
            nx, ny = float(toks[i]), float(toks[i + 1]); i += 2
            if cmd == 'l': x += nx; y += ny
            else: x, y = nx, ny
            cur['pts'].append((x, y))
        elif cmd in 'Hh':
            nx = float(toks[i]); i += 1
            x = x + nx if cmd == 'h' else nx
            cur['pts'].append((x, y))
        elif cmd in 'Vv':
            ny = float(toks[i]); i += 1
            y = y + ny if cmd == 'v' else ny
            cur['pts'].append((x, y))
        else:
            i += 1
    if cur is not None:
        subs.append(cur)

    for s in subs:
        pts = s['pts']
        a = 0.0
        for j in range(len(pts)):
            x1, y1 = pts[j]
            x2, y2 = pts[(j + 1) % len(pts)]
            a += x1 * y2 - x2 * y1
        s['area'] = a / 2.0
        ys = [p[1] for p in pts]
        s['ymin'], s['ymax'] = min(ys), max(ys)
    return subs


def fmt(v):
    """Compact number formatting (trim trailing zeros, no exponent)."""
    s = f'{v:.4f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


def split_subpath_strings(d):
    """Split the raw `d` into per-subpath substrings (each starts at a moveto)."""
    parts = [p for p in SUBPATH_SPLIT_RE.split(d) if p.strip()]
    return parts


def build_output(d, subs, keep):
    """Reassemble a `d` string from kept subpaths, all with absolute movetos."""
    parts = split_subpath_strings(d)
    if len(parts) != len(subs):
        raise RuntimeError(
            f'subpath count mismatch: {len(parts)} strings vs {len(subs)} parsed')
    out = []
    for raw, s, keep_it in zip(parts, subs, keep):
        if not keep_it:
            continue
        m = LEAD_MOVE_RE.match(raw)
        if not m:
            raise ValueError(f'bad subpath start: {raw[:40]!r}')
        rel = m.group(1) == 'm'
        rest = raw[m.end():]
        ax, ay = s['start']
        moveto = f'M{fmt(ax)} {fmt(ay)}'
        # If the remainder begins with implicit coords, make them explicit
        # linetos so the absolute M doesn't turn them into absolute L's.
        stripped = rest.lstrip()
        if stripped and (stripped[0].isdigit() or stripped[0] in '.+-'):
            lineto = 'l' if rel else 'L'
            out.append(moveto + lineto + rest)
        else:
            out.append(moveto + rest)
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', default='World_map_blank_without_borders.svg')
    ap.add_argument('-o', '--output', default='World_map_cleaned.svg')
    ap.add_argument('--min-island-area', type=float, default=100.0,
                    help='remove land polygons with |area| below this (px^2)')
    ap.add_argument('--min-lake-area', type=float, default=100.0,
                    help='remove (fill in) lake holes with |area| below this (px^2)')
    ap.add_argument('--antarctica-y', type=float, default=2050.0,
                    help='remove any subpath whose topmost point y exceeds this')
    ap.add_argument('--keep-antarctica', action='store_true',
                    help='do not remove Antarctica')
    args = ap.parse_args()

    svg = open(args.input, encoding='utf-8').read()
    dm = re.search(r'\bd="([^"]+)"', svg)
    if not dm:
        sys.exit('no <path d="..."> found')
    d = dm.group(1)

    subs = parse_subpaths(d)
    keep = [True] * len(subs)
    n_island = n_lake = n_antarctica = 0

    for idx, s in enumerate(subs):
        is_lake = s['area'] > 0          # positive winding = hole = lake
        area = abs(s['area'])
        in_antarctica = (not args.keep_antarctica) and s['ymin'] > args.antarctica_y
        if in_antarctica:
            keep[idx] = False
            n_antarctica += 1
        elif is_lake:
            if area < args.min_lake_area:
                keep[idx] = False
                n_lake += 1
        else:
            if area < args.min_island_area:
                keep[idx] = False
                n_island += 1

    new_d = build_output(d, subs, keep)
    new_svg = svg[:dm.start(1)] + new_d + svg[dm.end(1):]
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(new_svg)

    total = len(subs)
    kept = sum(keep)
    print(f'subpaths: {total} -> kept {kept} (removed {total - kept})')
    print(f'  small islands removed : {n_island}')
    print(f'  small lakes removed   : {n_lake}')
    print(f'  Antarctica removed    : {n_antarctica}')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
