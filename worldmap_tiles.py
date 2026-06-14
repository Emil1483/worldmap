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
from shapely.geometry import (LineString, Polygon as ShapelyPolygon,
                              box as ShapelyBox)
from shapely.ops import unary_union, nearest_points, split
from shapely.affinity import translate, rotate
from shapely import contains_xy
from scipy import ndimage

from worldmap_clean import parse_subpaths   # shared path parser


def fmt(v):
    s = f'{v:.2f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


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


def _parts(g):
    """A geometry's component Polygons (drops lines/points)."""
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [p for p in g.geoms if p.geom_type == 'Polygon']


def south_america(americas):
    """Split South America off the connected Americas at the THINNEST part -- the
    Panama isthmus. Erode the mass until it separates into a north and a south
    core, locate the neck between them, then cut straight across it (perpendicular
    to the neck axis). Falls back to a horizontal cut at the narrowest latitude
    if erosion never separates it. Returns the largest southern polygon, or None.
    """
    A = americas.area
    cores = None
    for d in np.arange(1.0, 300.0, 1.0):          # shrink until NA/SA pinch apart
        ps = [p for p in _parts(americas.buffer(-d)) if p.area > 0.10 * A]
        if len(ps) >= 2:
            cores = sorted(ps, key=lambda g: -g.area)[:2]
            break
    if cores is not None:
        north, south = sorted(cores, key=lambda g: g.centroid.y)  # small y = north
        pN, pS = nearest_points(north, south)     # closest approach across neck
        mx, my = (pN.x + pS.x) / 2, (pN.y + pS.y) / 2
        dx, dy = pS.x - pN.x, pS.y - pN.y
        L = math.hypot(dx, dy) or 1.0
        ux, uy = -dy / L, dx / L                   # unit vector perpendicular to neck
        bx = americas.bounds
        big = 3 * max(bx[2] - bx[0], bx[3] - bx[1])
        cut = LineString([(mx - ux * big, my - uy * big),
                          (mx + ux * big, my + uy * big)])
        pieces = [g for g in split(americas, cut).geoms
                  if g.geom_type == 'Polygon']
        sa = unary_union([g for g in pieces if g.intersection(south).area > 0])
        if not sa.is_empty:
            return max(_parts(sa), key=lambda g: g.area)
    # fallback: horizontal cut at the narrowest latitude
    bx = americas.bounds
    y0, y1 = bx[1], bx[3]
    neck = None
    for y in np.linspace(y0 + 0.30 * (y1 - y0), y0 + 0.70 * (y1 - y0), 200):
        w = americas.intersection(
            LineString([(bx[0] - 1, y), (bx[2] + 1, y)])).length
        if w > 0 and (neck is None or w < neck[1]):
            neck = (y, w)
    if neck is None:
        return None
    south = americas.intersection(
        ShapelyBox(bx[0] - 1, neck[0], bx[2] + 1, bx[3] + 1))
    parts = _parts(south)
    return max(parts, key=lambda g: g.area) if parts else None


def identify_copies(land):
    """Locate, in grid-mm space, the landmasses we want intact across a seam:
    Greenland (its own island, far north) and South America (south of the
    Panama neck of the connected Americas). Returns {name: polygon}."""
    comps = _parts(land)
    if not comps:
        return {}
    bx = land.bounds
    americas = min(comps, key=lambda g: g.bounds[0])   # the mass touching x=left
    north_band = bx[1] + 0.2 * (bx[3] - bx[1])
    arctic = [g for g in comps
              if g is not americas and g.centroid.y < north_band]
    out = {}
    sa = south_america(americas)
    if sa is not None and not sa.is_empty:
        out['South America'] = sa
    if arctic:                                          # biggest far-north island
        out['Greenland'] = max(arctic, key=lambda g: g.area)
    return out


def place_in_ocean(shape, blockers, cells, cell=5.0):
    """Find the most-open spot for `shape` inside a single grid cell's ocean,
    clear of everything in `blockers`. A copy must sit within one piece (so the
    laser cuts it whole from one sheet) and away from a seam. Tries 0/90 deg
    rotation and a shrinking clearance ring. Returns (placed_shape, (r, c)) or
    None if it will not fit any piece.

    `cells` is a list of (r, c, x0, y0, x1, y1) piece rectangles in mm.
    """
    total_w = max(c[4] for c in cells)
    total_h = max(c[5] for c in cells)
    W = int(math.ceil(total_w / cell))
    H = int(math.ceil(total_h / cell))
    gx, gy = np.meshgrid((np.arange(W) + 0.5) * cell, (np.arange(H) + 0.5) * cell)
    for clr in (12.0, 8.0, 4.0, 0.0):
        occ = np.zeros((H, W), bool)
        for b in blockers:
            occ |= contains_xy(b.buffer(clr) if clr else b, gx, gy)
        free = ~occ
        ii = np.zeros((H + 1, W + 1), np.int64)
        ii[1:, 1:] = np.cumsum(np.cumsum(free.astype(np.int64), 0), 1)
        dt = ndimage.distance_transform_edt(free)      # openness of each free cell
        best = None                                    # (dt_value, placed, rc)
        for ang in (0, 90):
            sh = rotate(shape, ang, origin='centroid') if ang else shape
            w = sh.bounds[2] - sh.bounds[0]
            h = sh.bounds[3] - sh.bounds[1]
            cw = int(math.ceil((w + 2 * clr) / cell))
            ch = int(math.ceil((h + 2 * clr) / cell))
            if cw > W or ch > H:
                continue
            # window-sum of free cells for every top-left position (vectorised)
            S = (ii[ch:, cw:] - ii[:-ch, cw:] - ii[ch:, :-cw] + ii[:-ch, :-cw])
            valid = S == cw * ch
            dtc = dt[ch // 2:ch // 2 + valid.shape[0],
                     cw // 2:cw // 2 + valid.shape[1]]
            for (r, c, x0, y0, x1, y1) in cells:
                ix0 = int(math.ceil(x0 / cell))
                iy0 = int(math.ceil(y0 / cell))
                ix1 = min(int(math.floor(x1 / cell)) - cw, valid.shape[1] - 1)
                iy1 = min(int(math.floor(y1 / cell)) - ch, valid.shape[0] - 1)
                if ix1 < ix0 or iy1 < iy0:
                    continue
                sv = valid[iy0:iy1 + 1, ix0:ix1 + 1]
                if not sv.any():
                    continue
                sd = np.where(sv, dtc[iy0:iy1 + 1, ix0:ix1 + 1], -1.0)
                k = np.unravel_index(int(np.argmax(sd)), sd.shape)
                v = sd[k]
                if v >= 0 and (best is None or v > best[0]):
                    r0, c0 = iy0 + k[0], ix0 + k[1]
                    cxm = (c0 + cw / 2) * cell
                    cym = (r0 + ch / 2) * cell
                    scx = (sh.bounds[0] + sh.bounds[2]) / 2
                    scy = (sh.bounds[1] + sh.bounds[3]) / 2
                    best = (v, translate(sh, cxm - scx, cym - scy), (r, c))
        if best:
            return best[1], best[2]
    return None


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
    ap.add_argument('--small-island-cm2', type=float, default=5.0,
                    help='a land island below this area that a seam would split '
                         'is kept whole on the piece holding its centre (cut '
                         'overhangs the edge) and dropped from the other piece')
    ap.add_argument('--border', action='store_true',
                    help='draw each piece outline rectangle (cut/registration)')
    ap.add_argument('--label', action='store_true',
                    help='engrave a small r#c# label in each piece corner')
    ap.add_argument('--no-copies', dest='copies', action='store_false',
                    help='do not place intact Greenland/South America copies in '
                         'the ocean (default: place them so a seam-free version '
                         'can be cut and glued over the seamed original)')
    ap.add_argument('--no-seam-edges', dest='seam_edges', action='store_false',
                    help='do not draw the vertical seam cut line where a piece '
                         'meets a neighbour (default: draw it)')
    ap.set_defaults(copies=True, seam_edges=True)
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
        return np.column_stack(((p[:, 0] - mnx) * scale + off_x,
                                (p[:, 1] - mny) * scale + off_y))

    # Build the land as a shapely geometry in grid-mm space: union the land
    # polygons, subtract the lakes (so holes stay holes). buffer(0) repairs the
    # self-touching rings the source is full of.
    land_parts, lake_parts = [], []
    for s, p in zip(subs, polys):
        poly = ShapelyPolygon(to_grid(p)).buffer(0)
        (land_parts if s['area'] < 0 else lake_parts).append(poly)
    land = unary_union(land_parts)
    if lake_parts:
        land = land.difference(unary_union(lake_parts))

    os.makedirs(args.outdir, exist_ok=True)

    # Split the land at the seams into per-piece connected lobes. A lobe that is
    # small (< --small-island-cm2) and is NOT its landmass's biggest fragment is
    # an island the cut would orphan: move it WHOLE to the neighbouring piece
    # that holds the body it was attached to (overhang past the edge — sheet is
    # oversized), and drop it here. Big landmasses still split at the seam.
    thr = args.small_island_cm2 * 100.0       # mm^2
    box = {}
    for r in range(rows):
        for c in range(cols):
            box[(r, c)] = ShapelyPolygon([(c * pw, r * ph), ((c + 1) * pw, r * ph),
                                          ((c + 1) * pw, (r + 1) * ph),
                                          (c * pw, (r + 1) * ph)])

    def polygons(g):
        if g.is_empty:
            return []
        if g.geom_type == 'Polygon':
            return [g]
        return [p for p in g.geoms if p.geom_type == 'Polygon']

    keep = {(r, c): [] for r in range(rows) for c in range(cols)}
    moved_lobes = []                          # islands kept whole (cut detours here)
    n_moved = 0
    for mass in polygons(land):               # each connected landmass
        frag = {rc: mass.intersection(b) for rc, b in box.items()}
        frag = {rc: g for rc, g in frag.items() if not g.is_empty}
        body_rc = max(frag, key=lambda rc: frag[rc].area)
        for rc, g in frag.items():
            r, c = rc
            for lobe in polygons(g):
                # neighbours this lobe reaches across an interior seam
                minx, miny, maxx, maxy = lobe.bounds
                nbrs = []
                if c > 0 and minx <= c * pw + 1e-6:
                    nbrs.append((r, c - 1))
                if c < cols - 1 and maxx >= (c + 1) * pw - 1e-6:
                    nbrs.append((r, c + 1))
                if r > 0 and miny <= r * ph + 1e-6:
                    nbrs.append((r - 1, c))
                if r < rows - 1 and maxy >= (r + 1) * ph - 1e-6:
                    nbrs.append((r + 1, c))
                orphan = (lobe.area < thr and rc != body_rc and nbrs and
                          lobe.area < mass.area)
                if orphan:                    # hand to the biggest adjacent piece
                    owner = max(nbrs, key=lambda n: frag.get(n, lobe).area
                                if n in frag else 0.0)
                    keep[owner].append(lobe)
                    moved_lobes.append(lobe)
                    n_moved += 1
                else:
                    keep[rc].append(lobe)

    coverage = scale * scale * map_w * map_h / (cols * rows * pw * ph)
    print(f'map aspect {map_aspect:.3f} -> grid {cols}x{rows} of '
          f'{pw/10:.0f}x{ph/10:.0f}cm  (scale {scale:.4f}, '
          f'wood used {coverage*100:.0f}%)'
          + (f', {n_moved} small island(s) kept whole + overcut' if n_moved else ''))

    def ring_d(coords, x0, y0):
        pts = list(coords)
        seg = ['M', fmt(pts[0][0] - x0), fmt(pts[0][1] - y0), 'L']
        for x, y in pts[1:]:
            seg.append(fmt(x - x0))
            seg.append(fmt(y - y0))
        seg.append('Z')
        return ' '.join(seg)

    def geom_d(g, x0, y0):                     # (multi)polygon -> SVG path data
        parts = []
        for poly in polygons(g):
            parts.append(ring_d(poly.exterior.coords, x0, y0))
            for hole in poly.interiors:
                parts.append(ring_d(hole.coords, x0, y0))
        return ''.join(parts)

    # Intact copies: drop a seam-free Greenland / South America into open ocean
    # so the user can cut a whole one and glue it over the seamed original. Each
    # copy must live inside a single piece (one sheet, away from a seam); any
    # that won't fit any piece's ocean go onto a separate patches.svg.
    copies_placed = []                         # [(name, polygon, (r, c)), ...]
    patches = []                               # [(name, polygon), ...] not placed
    if args.copies:
        cells = [(r, c, c * pw, r * ph, (c + 1) * pw, (r + 1) * ph)
                 for r in range(rows) for c in range(cols)]
        for name, shp in identify_copies(land).items():
            blockers = [land] + [g for _, g, _ in copies_placed]
            res = place_in_ocean(shp, blockers, cells)
            if res:
                moved, rc = res
                keep[rc].append(moved)
                copies_placed.append((name, moved, rc))
            else:
                patches.append((name, shp))
        for name, _, rc in copies_placed:
            print(f'  intact {name} copy -> piece r{rc[0]+1}c{rc[1]+1} ocean')
        for name, _ in patches:
            print(f'  intact {name} copy did not fit any piece ocean '
                  f'-> patches.svg')

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * pw, r * ph
            # union the kept lobes so an orphan merges with its body (no re-cut)
            g = unary_union(keep[(r, c)]) if keep[(r, c)] else None
            d = geom_d(g, x0, y0) if g is not None and not g.is_empty else ''
            # canvas grows to include any overhang past the nominal piece edge
            if d:
                gx0, gy0, gx1, gy1 = g.bounds
                vx0, vy0 = min(0.0, gx0 - x0), min(0.0, gy0 - y0)
                vx1, vy1 = max(pw, gx1 - x0), max(ph, gy1 - y0)
            else:
                vx0, vy0, vx1, vy1 = 0.0, 0.0, pw, ph
            body = ''
            if d:
                body += f'<path fill="{fill}" fill-rule="evenodd" d="{d}"/>'
            if args.seam_edges:
                # cut line along the seam where this piece abuts a neighbour, but
                # GAPPED wherever land overhangs past the seam (kept-whole islands
                # / out-of-bounds growth) -- there the coastline is the real cut,
                # so a straight seam line would slice off the intentional overhang.
                box_rect = ShapelyBox(x0, y0, x0 + pw, y0 + ph)
                over = (g.difference(box_rect).buffer(0.5)
                        if g is not None and not g.is_empty else None)
                for X, draw in ((x0, c > 0), (x0 + pw, c < cols - 1)):
                    if not draw:
                        continue
                    seg = LineString([(X, y0), (X, y0 + ph)])
                    if over is not None and not over.is_empty:
                        seg = seg.difference(over)
                    if seg.is_empty:
                        continue
                    for ls in (seg.geoms if seg.geom_type.startswith('Multi')
                               else [seg]):
                        if ls.geom_type != 'LineString' or ls.is_empty:
                            continue
                        ys = [p[1] for p in ls.coords]
                        body += (f'<line x1="{fmt(X - x0)}" '
                                 f'y1="{fmt(min(ys) - y0)}" x2="{fmt(X - x0)}" '
                                 f'y2="{fmt(max(ys) - y0)}" '
                                 f'stroke="#f00" stroke-width="0.1"/>')
            if args.border:
                body += (f'<rect x="0" y="0" width="{fmt(pw)}" height="{fmt(ph)}" '
                         f'fill="none" stroke="#f00" stroke-width="0.5"/>')
            if args.label:
                body += (f'<text x="6" y="20" font-size="14" '
                         f'fill="#00f">r{r+1}c{c+1}</text>')
            out = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                   f'width="{fmt(vx1 - vx0)}mm" height="{fmt(vy1 - vy0)}mm" '
                   f'viewBox="{fmt(vx0)} {fmt(vy0)} {fmt(vx1 - vx0)} {fmt(vy1 - vy0)}" '
                   f'version="1.0">{body}</svg>')
            path = os.path.join(args.outdir, f'piece_r{r+1}_c{c+1}.svg')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(out)

    # assembly overview: full map + the ACTUAL cut lines + labels. The red cut
    # follows each seam, but breaks where a kept-whole island crosses it and
    # detours along that island's coastline (so the island stays on one piece).
    moved = unary_union(moved_lobes) if moved_lobes else None
    seam_lines = ([LineString([(c * pw, 0), (c * pw, total_h)]) for c in range(1, cols)] +
                  [LineString([(0, r * ph), (total_w, r * ph)]) for r in range(1, rows)])
    all_seams = unary_union(seam_lines) if seam_lines else None

    def line_svg(geom):
        out = []
        for ls in (geom.geoms if geom.geom_type.startswith('Multi') else [geom]):
            if ls.geom_type != 'LineString' or ls.is_empty:
                continue
            pts = ' '.join(f'{fmt(x)},{fmt(y)}' for x, y in ls.coords)
            out.append(f'<polyline points="{pts}" fill="none" stroke="#f00" '
                       f'stroke-width="2"/>')
        return ''.join(out)

    ov = [f'<svg xmlns="http://www.w3.org/2000/svg" '
          f'width="{fmt(total_w)}mm" height="{fmt(total_h)}mm" '
          f'viewBox="0 0 {fmt(total_w)} {fmt(total_h)}" version="1.0">']
    ov.append(f'<path fill="{fill}" fill-rule="evenodd" d="{geom_d(land, 0, 0)}"/>')
    for name, g, rc in copies_placed:          # intact copies dropped in the ocean
        ov.append(f'<path fill="{fill}" fill-rule="evenodd" d="{geom_d(g, 0, 0)}"/>')
        cx, cy = g.centroid.x, g.centroid.y
        ov.append(f'<text x="{fmt(cx)}" y="{fmt(cy)}" font-size="16" '
                  f'text-anchor="middle" fill="#080">{name} (copy)</text>')
    ov.append(f'<rect x="0" y="0" width="{fmt(total_w)}" height="{fmt(total_h)}" '
              f'fill="none" stroke="#f00" stroke-width="2"/>')   # outer sheet edges
    for seam in seam_lines:                    # interior seams, gapped at kept islands
        ov.append(line_svg(seam.difference(moved) if moved is not None else seam))
    if moved is not None:                      # detour: island coastline = the real cut
        ov.append(line_svg(moved.boundary.difference(all_seams)))
    for r in range(rows):
        for c in range(cols):
            ov.append(f'<text x="{fmt(c*pw+20)}" y="{fmt(r*ph+50)}" '
                      f'font-size="40" fill="#00f">r{r+1}c{c+1}</text>')
    ov.append('</svg>')
    with open(os.path.join(args.outdir, 'index.svg'), 'w', encoding='utf-8') as f:
        f.write(''.join(ov))

    # Copies too big for any piece's ocean go on their own spare sheet, stacked
    # vertically in real mm so they can be cut and glued over the seamed version.
    if patches:
        m = 20.0
        yc, sheet_w, parts = m, 0.0, []
        for name, shp in patches:
            b = shp.bounds
            g = translate(shp, m - b[0], yc - b[1])
            parts.append(f'<path fill="{fill}" fill-rule="evenodd" '
                         f'd="{geom_d(g, 0, 0)}"/>')
            parts.append(f'<text x="{fmt(m)}" y="{fmt(yc - 6)}" font-size="10" '
                         f'fill="#080">{name} (copy)</text>')
            sheet_w = max(sheet_w, b[2] - b[0])
            yc += (b[3] - b[1]) + m
        sheet_w += 2 * m
        patch_svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                     f'width="{fmt(sheet_w)}mm" height="{fmt(yc)}mm" '
                     f'viewBox="0 0 {fmt(sheet_w)} {fmt(yc)}" version="1.0">'
                     f'{"".join(parts)}</svg>')
        with open(os.path.join(args.outdir, 'patches.svg'), 'w',
                  encoding='utf-8') as f:
            f.write(patch_svg)

    print(f'wrote {cols*rows} pieces + index.svg'
          + (' + patches.svg' if patches else '') + f' to {args.outdir}/')


if __name__ == '__main__':
    main()
