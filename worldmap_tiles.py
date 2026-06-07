#!/usr/bin/env python3
"""Tile a projected world-map SVG across N rectangular wood pieces for laser cut.

Pipeline stage 3 (after worldmap_clean -> worldmap_project). Given a number of
physical wood pieces (default 100cm x 60cm each), this:

  1. picks the piece grid (cols x rows, using exactly N pieces) whose combined
     aspect ratio best matches the map, trying both piece orientations;
  2. scales the map as large as possible to fit that grid (preserving aspect),
     centred, so adjacent pieces line up seam-to-seam;
  3. geometrically clips every land polygon to each piece's rectangle
     (Sutherland-Hodgman) so each output SVG contains only its own slice;
  4. writes one SVG per piece, sized in real millimetres so the laser cutter
     gets true physical dimensions, plus an index.svg assembly overview.

Output files: <outdir>/piece_r{row}_c{col}.svg and <outdir>/index.svg.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

from worldmap_clean import parse_subpaths   # shared path parser


def fmt(v):
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


# ---- Sutherland-Hodgman polygon clip against an axis-aligned rectangle ------

def clip_poly(pts, xmin, ymin, xmax, ymax):
    """Clip closed polygon `pts` (list of (x,y)) to the rectangle. Returns a
    list of (x,y) for the clipped polygon, or [] if nothing is inside."""
    def clip_edge(poly, inside, intersect):
        if not poly:
            return poly
        out = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def isect(p, q, t):
        return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

    poly = list(pts)
    # left
    poly = clip_edge(poly, lambda p: p[0] >= xmin,
                     lambda p, q: isect(p, q, (xmin - p[0]) / (q[0] - p[0])))
    # right
    poly = clip_edge(poly, lambda p: p[0] <= xmax,
                     lambda p, q: isect(p, q, (xmax - p[0]) / (q[0] - p[0])))
    # top
    poly = clip_edge(poly, lambda p: p[1] >= ymin,
                     lambda p, q: isect(p, q, (ymin - p[1]) / (q[1] - p[1])))
    # bottom
    poly = clip_edge(poly, lambda p: p[1] <= ymax,
                     lambda p, q: isect(p, q, (ymax - p[1]) / (q[1] - p[1])))
    return poly


def choose_grid(n, pw, ph, map_aspect):
    """Pick (cols, rows, piece_w, piece_h) using exactly n pieces whose grid
    aspect best matches map_aspect. Tries both piece orientations."""
    best = None
    for (a, b) in ((pw, ph), (ph, pw)):          # piece orientation
        for cols in range(1, n + 1):
            if n % cols:
                continue
            rows = n // cols
            grid_aspect = (cols * a) / (rows * b)
            score = abs(math.log(grid_aspect / map_aspect))
            if best is None or score < best[0]:
                best = (score, cols, rows, a, b)
    _, cols, rows, a, b = best
    return cols, rows, a, b


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', default='World_map_gall.svg')
    ap.add_argument('-n', '--pieces', type=int, required=True,
                    help='number of wood pieces (= number of output SVGs)')
    ap.add_argument('-o', '--outdir', default='wood_pieces')
    ap.add_argument('--piece-width-cm', type=float, default=100.0)
    ap.add_argument('--piece-height-cm', type=float, default=60.0)
    ap.add_argument('--margin-mm', type=float, default=0.0,
                    help='blank margin kept around the whole map inside the grid')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='0-1 fraction of the max fit-to-grid scale (1 = as large '
                         'as fits)')
    ap.add_argument('--align-horizontal', type=float, default=0.5,
                    help='0-1 horizontal placement of the map in spare width '
                         '(0 = flush left, 1 = flush right, 0.5 = centred)')
    ap.add_argument('--border', action='store_true',
                    help='draw each piece outline rectangle (cut/registration)')
    ap.add_argument('--label', action='store_true',
                    help='engrave a small r#c# label in each piece corner')
    args = ap.parse_args()

    if args.pieces < 1:
        sys.exit('--pieces must be >= 1')
    if not 0 < args.scale <= 1:
        sys.exit('--scale must be in (0, 1]')
    if not 0 <= args.align_horizontal <= 1:
        sys.exit('--align-horizontal must be in [0, 1]')

    svg = open(args.input, encoding='utf-8').read()
    dm = re.search(r'\bd="([^"]+)"', svg)
    if not dm:
        sys.exit('no <path d="..."> found')
    subs = parse_subpaths(dm.group(1))
    fillm = re.search(r'fill="([^"]+)"', svg)
    fill = fillm.group(1) if fillm else '#bcbcbc'

    polys = [np.asarray(s['pts'], dtype=float) for s in subs]
    allp = np.concatenate(polys)
    mnx, mxx = allp[:, 0].min(), allp[:, 0].max()
    mny, mxy = allp[:, 1].min(), allp[:, 1].max()
    map_w, map_h = mxx - mnx, mxy - mny
    map_aspect = map_w / map_h

    # physical piece + grid, in mm
    pw, ph = args.piece_width_cm * 10.0, args.piece_height_cm * 10.0
    cols, rows, pw, ph = choose_grid(args.pieces, pw, ph, map_aspect)
    total_w, total_h = cols * pw, rows * ph

    # fit map into the usable area (grid minus margin), preserving aspect, then
    # shrink by --scale and place horizontally by --align-horizontal
    avail_w, avail_h = total_w - 2 * args.margin_mm, total_h - 2 * args.margin_mm
    scale = min(avail_w / map_w, avail_h / map_h) * args.scale
    draw_w, draw_h = map_w * scale, map_h * scale
    # spare width gets distributed left/right by align (0 = left, 1 = right)
    off_x = args.margin_mm + args.align_horizontal * max(0.0, avail_w - draw_w)
    off_y = (total_h - draw_h) / 2.0          # vertical: always centred

    def to_grid(p):                            # source px -> grid mm
        return ((p[:, 0] - mnx) * scale + off_x,
                (p[:, 1] - mny) * scale + off_y)

    grid_polys = []
    for p in polys:
        gx, gy = to_grid(p)
        grid_polys.append(list(zip(gx.tolist(), gy.tolist())))

    os.makedirs(args.outdir, exist_ok=True)

    coverage = scale * scale * map_w * map_h / (cols * rows * pw * ph)
    print(f'map aspect {map_aspect:.3f} -> grid {cols}x{rows} of '
          f'{pw/10:.0f}x{ph/10:.0f}cm  (scale {scale:.4f}, '
          f'wood used {coverage*100:.0f}%)')

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * pw, r * ph
            x1, y1 = x0 + pw, y0 + ph
            parts = []
            for poly in grid_polys:
                xs = [q[0] for q in poly]
                ys = [q[1] for q in poly]
                if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
                    continue                   # bbox reject
                cp = clip_poly(poly, x0, y0, x1, y1)
                if len(cp) < 3:
                    continue
                seg = ['M', fmt(cp[0][0] - x0), fmt(cp[0][1] - y0), 'L']
                for q in cp[1:]:
                    seg.append(fmt(q[0] - x0))
                    seg.append(fmt(q[1] - y0))
                seg.append('Z')
                parts.append(' '.join(seg))

            body = ''
            if parts:
                body += f'<path fill="{fill}" fill-rule="nonzero" d="{"".join(parts)}"/>'
            if args.border:
                body += (f'<rect x="0" y="0" width="{fmt(pw)}" height="{fmt(ph)}" '
                         f'fill="none" stroke="#f00" stroke-width="0.5"/>')
            if args.label:
                body += (f'<text x="6" y="20" font-size="14" '
                         f'fill="#00f">r{r+1}c{c+1}</text>')

            out = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'width="{fmt(pw)}mm" height="{fmt(ph)}mm" '
                   f'viewBox="0 0 {fmt(pw)} {fmt(ph)}" version="1.0">{body}</svg>')
            path = os.path.join(args.outdir, f'piece_r{r+1}_c{c+1}.svg')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(out)

    # assembly overview: full map + grid lines + labels, in grid-mm space
    ov = [f'<svg xmlns="http://www.w3.org/2000/svg" '
          f'width="{fmt(total_w)}mm" height="{fmt(total_h)}mm" '
          f'viewBox="0 0 {fmt(total_w)} {fmt(total_h)}" version="1.0">']
    full = []
    for poly in grid_polys:
        full.append('M' + fmt(poly[0][0]) + ' ' + fmt(poly[0][1]) + 'L'
                     + ' '.join(f'{fmt(q[0])} {fmt(q[1])}' for q in poly[1:]) + 'Z')
    ov.append(f'<path fill="{fill}" fill-rule="nonzero" d="{"".join(full)}"/>')
    for r in range(rows):
        for c in range(cols):
            ov.append(f'<rect x="{fmt(c*pw)}" y="{fmt(r*ph)}" width="{fmt(pw)}" '
                      f'height="{fmt(ph)}" fill="none" stroke="#f00" '
                      f'stroke-width="2"/>')
            ov.append(f'<text x="{fmt(c*pw+20)}" y="{fmt(r*ph+50)}" '
                      f'font-size="40" fill="#00f">r{r+1}c{c+1}</text>')
    ov.append('</svg>')
    with open(os.path.join(args.outdir, 'index.svg'), 'w', encoding='utf-8') as f:
        f.write(''.join(ov))

    print(f'wrote {cols*rows} pieces + index.svg to {args.outdir}/')


if __name__ == '__main__':
    main()
