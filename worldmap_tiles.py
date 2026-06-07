#!/usr/bin/env python3
"""Tile a projected world-map SVG across N rectangular wood pieces for laser cut.

Pipeline stage 3 (after worldmap_clean -> worldmap_project). Given a number of
physical wood pieces (default 100cm x 60cm each), this:

  1. picks the piece grid (cols x rows, using exactly N pieces) whose combined
     aspect ratio best matches the map, trying both piece orientations;
  2. scales the map as large as possible to fit that grid (preserving aspect),
     then slides/shrinks it so no seam shaves a thin coastal sliver of wood off
     a landmass (a fragment below --min-fragment-cm2 touching a seam);
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
from PIL import Image, ImageDraw
from scipy import ndimage

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


# ---- area of a polygon on one side of an axis-aligned cut line -------------

def _shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_halfplane(P, ci, X):
    """Keep the part of polygon P whose coordinate `ci` is <= X."""
    out = []
    n = len(P)
    for i in range(n):
        a = P[i]
        b = P[(i + 1) % n]
        a_in = a[ci] <= X
        b_in = b[ci] <= X
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (X - a[ci]) / (b[ci] - a[ci])
            out.append(a + t * (b - a))
    return np.asarray(out) if len(out) >= 3 else None


def area_profile(P, ci, n_samples=200):
    """Sample the area of P lying on the `ci <= X` side as the cut line X
    sweeps across the polygon's extent. Because sweeping the line only ever
    *adds* area to the near side, the result is monotonic non-decreasing —
    so the cut position for any target area is a single np.interp lookup.

    Returns (xs, area_le, total)."""
    lo, hi = P[:, ci].min(), P[:, ci].max()
    xs = np.linspace(lo, hi, n_samples)
    al = np.array([0.0 if (c := _clip_halfplane(P, ci, X)) is None
                   else _shoelace(c) for X in xs])
    return xs, np.maximum.accumulate(al), al.max()


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
    ap.add_argument('--min-fragment-cm2', type=float, default=5.0,
                    help='a seam that would shave a wood fragment smaller than '
                         'this off a landmass is treated as an illegal cut; the '
                         'map is slid/scaled so no seam lands in such a zone')
    ap.add_argument('--seam-clearance-mm', type=float, default=1.0,
                    help='keep seams at least this far from an illegal cut zone')
    ap.add_argument('--no-protect', action='store_true',
                    help='disable sliver protection (use full max-fit scale)')
    ap.add_argument('--border', action='store_true',
                    help='draw each piece outline rectangle (cut/registration)')
    ap.add_argument('--label', action='store_true',
                    help='engrave a small r#c# label in each piece corner')
    args = ap.parse_args()

    if args.pieces < 1:
        sys.exit('--pieces must be >= 1')

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

    # fit map into the usable area (grid minus margin), preserving aspect
    avail_w, avail_h = total_w - 2 * args.margin_mm, total_h - 2 * args.margin_mm
    max_scale = min(avail_w / map_w, avail_h / map_h)

    # Sliver mitigation. A seam that clips just past a coastline shaves a thin
    # fragment off a landmass, leaving a fragile/floating chip on one piece. We
    # slide the map (free — no size loss) so seams miss such spots, shrinking
    # only if forced, and pick the placement by the *true* test: clip every
    # landmass to every piece and count fragments below --min-fragment that
    # touch an interior seam (count_slivers).
    #
    # That true test is too costly to search blindly, so it's driven by a fast
    # proposal: per axis, the "illegal cut zones" are the source-x ranges where a
    # vertical cut shaves off < a fragment of a landmass. Because sweeping a cut
    # line only ever adds area to one side, area-on-one-side is monotonic, so the
    # zones are just the two extremities of each landmass (a peninsula is an
    # extremity zone reaching inland). candidate_offsets() proposes offsets in
    # the gaps between zones; count_slivers() is the gate that decides — it also
    # catches tongues mid-landmass that the per-axis extremity model can't see.
    land = [p for s, p in zip(subs, polys) if s['area'] < 0]

    def decimate(P, cap=1500):                 # area only needs coarse outline
        return P if len(P) <= cap else P[::int(math.ceil(len(P) / cap))]

    prof_x = [area_profile(decimate(P), 0) for P in land]
    prof_y = [area_profile(decimate(P), 1) for P in land]

    def illegal_zones(profiles, scale):
        """Source-coord intervals where a cut would leave a sub-fragment piece."""
        thr = args.min_fragment_cm2 * 100.0 / (scale * scale)   # mm^2 -> source px^2
        zones = []
        for xs, al, total in profiles:
            if total <= 2 * thr:               # whole landmass below 2 fragments
                zones.append((xs[0], xs[-1]))  # any cut here makes a sub-fragment
            else:
                xa = float(np.interp(thr, al, xs))         # left side == thr
                xb = float(np.interp(total - thr, al, xs)) # right side == thr
                zones.append((xs[0], xa))
                zones.append((xb, xs[-1]))
        return zones

    def candidate_offsets(scale, axis, k=8):
        """Up to k in-bounds offsets whose seams land in the fewest illegal
        zones (the gaps between zones). The extremity-area model is blind to
        tongues mid-landmass, so these are *proposals* — the true per-cell
        fragment area (count_slivers) is the gate that decides between them."""
        if axis == 'x':
            profiles, total, pitch, ncuts, span = prof_x, total_w, pw, cols, map_w
        else:
            profiles, total, pitch, ncuts, span = prof_y, total_h, ph, rows, map_h
        clr = args.seam_clearance_mm
        olo = args.margin_mm
        ohi = total - span * scale - args.margin_mm
        if ohi < olo:                          # axis fills the grid: no slide room
            return [(total - span * scale) / 2.0]
        seams = [j * pitch for j in range(1, ncuts)]
        forb = []
        for z0, z1 in illegal_zones(profiles, scale):
            for sm in seams:                   # off in [sm-z1*scale, sm-z0*scale]
                forb.append((sm - z1 * scale - clr, sm - z0 * scale + clr))
        centre = (total - span * scale) / 2.0
        cands = {olo, ohi, centre}
        for f0, f1 in forb:                    # just outside each zone edge
            for v in (f0 - 1e-6, f1 + 1e-6):
                if olo <= v <= ohi:
                    cands.add(v)
        scored = sorted(cands, key=lambda o: (sum(1 for f0, f1 in forb if f0 < o < f1),
                                              abs(o - centre)))
        return scored[:k]

    # --- true gate: count floating chips a seam would orphan -----------------
    # Summed polygon∩piece area is NOT enough: clipping bridges separate lobes of
    # one landmass along the seam edge, hiding a small floating lobe behind a big
    # chunk of the same polygon. The honest test is physical connectivity. We
    # rasterise the land with a per-landmass id, slice along every interior seam,
    # label connected components, and flag a component when it is small
    # (< --min-fragment) yet belongs to a landmass that is large — i.e. the seam
    # orphaned a chip off a continent. A naturally small island grazing a seam is
    # not flagged (its landmass was already small); a continent split into two
    # big halves is fine (neither half is small).
    thr_mm2 = args.min_fragment_cm2 * 100.0
    RES = 0.5                                   # raster px per mm (2mm grid: ample
    #                                             for the cm-scale chips we hunt)

    # Rasterise ONCE in source space, each land polygon stamped with its own id
    # (lakes punched to 0). Sliding/scaling moves only the seams relative to the
    # fixed land, so we never redraw — per scale we resize, per offset move cuts.
    land_polys = [P for s, P in zip(subs, polys) if s['area'] < 0]
    n_land = len(land_polys)
    src_img = Image.new('I', (max(1, int(math.ceil(map_w))),
                              max(1, int(math.ceil(map_h)))), 0)
    src_dr = ImageDraw.Draw(src_img)
    for pid, P in enumerate(land_polys, start=1):
        src_dr.polygon(np.column_stack((P[:, 0] - mnx, P[:, 1] - mny)).ravel().tolist(),
                       fill=pid)
    for s, P in zip(subs, polys):               # punch lakes back to sea
        if s['area'] >= 0:
            src_dr.polygon(np.column_stack((P[:, 0] - mnx, P[:, 1] - mny)).ravel().tolist(),
                           fill=0)

    def land_raster(scale):
        rw = max(1, int(round(map_w * scale * RES)))
        rh = max(1, int(round(map_h * scale * RES)))
        ids = np.asarray(src_img.resize((rw, rh), Image.NEAREST)).astype(np.int32)
        full = np.bincount(ids.ravel(), minlength=n_land + 1)   # drawn area per id
        return ids, full, rw, rh

    def count_slivers(ids, full, rw, rh, ox, oy):
        cut_c = [c for c in (int(round((j * pw - ox) * RES)) for j in range(1, cols))
                 if 0 <= c < rw]
        cut_r = [r for r in (int(round((j * ph - oy) * RES)) for j in range(1, rows))
                 if 0 <= r < rh]
        if not cut_c and not cut_r:
            return 0
        a = ids.copy()
        for col in cut_c:                       # a zeroed line fully severs both sides
            a[:, col] = 0
        for row in cut_r:
            a[row, :] = 0
        lab, nlab = ndimage.label(a > 0)        # 4-connectivity
        if nlab == 0:
            return 0
        sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
        thr_px = thr_mm2 * RES * RES
        touch = set()                           # component labels hugging a seam
        for col in cut_c:
            for sx in (col - 1, col + 1):
                if 0 <= sx < rw:
                    touch.update(lab[:, sx][lab[:, sx] > 0].tolist())
        for row in cut_r:
            for sy in (row - 1, row + 1):
                if 0 <= sy < rh:
                    touch.update(lab[sy, :][lab[sy, :] > 0].tolist())
        small = [L for L in touch if 0 < sizes[L] < thr_px]
        if not small:
            return 0
        # poly id is constant within a component -> its full landmass area
        pid = np.asarray(ndimage.maximum(ids, lab, index=small)).astype(int)
        return int((full[pid] >= thr_px).sum())   # orphaned chip of a big landmass

    scale = max_scale
    off_x = (total_w - map_w * max_scale) / 2.0    # default: centred at max-fit
    off_y = (total_h - map_h * max_scale) / 2.0
    if not args.no_protect:
        best, s, stall = None, max_scale, 0
        while s > max_scale * 0.6:
            ids_ref, full_ref, rw, rh = land_raster(s)   # one resize per scale
            xs = candidate_offsets(s, 'x', k=5)
            ys = candidate_offsets(s, 'y', k=5)
            cx = (total_w - map_w * s) / 2.0
            cy = (total_h - map_h * s) / 2.0
            prev = best[0] if best else None
            # evaluate the true metric over the proposal grid; prefer fewest
            # slivers, then larger scale (seen first), then nearest centre
            for ox in xs:
                for oy in ys:
                    nsl = count_slivers(ids_ref, full_ref, rw, rh, ox, oy)
                    key = (nsl, abs(ox - cx) + abs(oy - cy))
                    if best is None or (key[0], -s, key[1]) < (best[0], -best[1], best[4]):
                        best = (nsl, s, ox, oy, key[1])
                    if nsl == 0:
                        break
                if best[0] == 0 and best[1] == s:
                    break
            if best[0] == 0:
                break
            # stop shrinking once it stops reducing the residual (it usually won't)
            stall = 0 if (prev is None or best[0] < prev) else stall + 1
            if stall >= 2:
                break
            s *= 0.95
        nsl, scale, off_x, off_y = best[0], best[1], best[2], best[3]
        shrunk = scale < max_scale - 1e-9
        thr = f'{args.min_fragment_cm2:g}cm2'
        if nsl == 0 and shrunk:
            print(f'sliver protection: slid + scaled {max_scale:.4f} -> {scale:.4f} '
                  f'so no seam shaves a fragment < {thr}')
        elif nsl == 0:
            print(f'sliver protection: slid map (no size loss) so no seam shaves '
                  f'a fragment < {thr}')
        else:
            print(f'sliver protection: scaled {max_scale:.4f} -> {scale:.4f}, '
                  f'minimised to {nsl} unavoidable sliver(s) < {thr} '
                  f'(raise --seam-clearance-mm, lower --min-fragment-cm2, or try '
                  f'a different -n)', file=sys.stderr)

    def to_grid(p):                            # source px -> grid mm (off_x/off_y set above)
        return ((p[:, 0] - mnx) * scale + off_x,
                (p[:, 1] - mny) * scale + off_y)

    grid_polys = []
    for p in polys:
        gx, gy = to_grid(p)
        grid_polys.append(list(zip(gx.tolist(), gy.tolist())))

    os.makedirs(args.outdir, exist_ok=True)
    # clear stale pieces from a previous run (e.g. larger n) so the outdir only
    # ever holds the current run's output
    for f in os.listdir(args.outdir):
        if re.fullmatch(r'piece_r\d+_c\d+\.svg|index\.svg', f):
            os.remove(os.path.join(args.outdir, f))

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
