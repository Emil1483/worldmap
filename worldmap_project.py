#!/usr/bin/env python3
"""Reproject the equirectangular world-map SVG into a classical projection,
with an optional coastline low-pass filter (--max-fjord-length).

Source projection (determined empirically, RMS ~1px against known capes from
84N to 54S) is a *scaled equirectangular* / plate carree:

    lon = x / WIDTH * 360 - 180                 (12.16 px/deg, full +-180)
    lat = (LAT_B - y) / LAT_M                    (14.40 px/deg)

so every pixel inverts cleanly to lon/lat. Each subpath is converted to
lon/lat, optionally smoothed, then forward-projected into the chosen target
and scaled to fit the output canvas. Winding order is preserved, so lakes
(holes) stay holes under the nonzero fill rule.

Projections implemented (all "classical world map" looking):
  equalearth  - Equal Earth (equal-area): Greenland exactly the right size.
  mollweide   - Mollweide (equal-area, elliptical).
  robinson    - Robinson (compromise; the textbook/NatGeo look).
  winkel      - Winkel Tripel (compromise; NatGeo standard).

The fjord filter is a Gaussian low-pass on the coastline in arc-length space
(km), applied in lon/lat before projection: inlets/peninsulas whose along-
coast length is below ~--max-fjord-length are smoothed away (e.g. Norway's
fjords), while large bays are kept.
"""

import argparse
import math
import re
import sys

import numpy as np

# ---- source equirectangular calibration -----------------------------------
WIDTH = 4378.13
LAT_M = 14.4002      # px per degree latitude
LAT_B = 1203.714     # y at the equator

DEG = math.pi / 180.0
R_EARTH_KM = 6371.0

from worldmap_clean import parse_subpaths   # shared path parser


def inv_lonlat(x, y):
    """Source pixel -> (lon, lat) in degrees."""
    lon = x / WIDTH * 360.0 - 180.0
    lat = (LAT_B - y) / LAT_M
    return lon, lat


# ---- forward projections (input lon/lat in RADIANS, output planar X,Y) -----

def p_equalearth(lon, lat):
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    th = np.arcsin(math.sqrt(3) / 2.0 * np.sin(lat))
    th2 = th * th
    den = 9 * A4 * th2**4 + 7 * A3 * th2**3 + 3 * A2 * th2 + A1
    x = 2 * math.sqrt(3) * lon * np.cos(th) / (3 * den)
    y = A1 * th + A2 * th**3 + A3 * th**7 + A4 * th**9
    return x, y


def p_mollweide(lon, lat):
    # solve 2*psi + sin(2*psi) = pi*sin(lat) by Newton iteration
    psi = lat.copy()
    s = math.pi * np.sin(lat)
    for _ in range(20):
        psi = psi - (2 * psi + np.sin(2 * psi) - s) / (2 + 2 * np.cos(2 * psi))
    # poles: derivative ->0; clamp
    psi = np.where(np.abs(np.abs(lat) - math.pi / 2) < 1e-9,
                   np.sign(lat) * math.pi / 2, psi)
    x = (2 * math.sqrt(2) / math.pi) * lon * np.cos(psi)
    y = math.sqrt(2) * np.sin(psi)
    return x, y


# Robinson lookup table: latitude(deg) -> (parallel-length, y-distance)
_ROB = np.array([
    [0,  1.0000, 0.0000], [5,  0.9986, 0.0620], [10, 0.9954, 0.1240],
    [15, 0.9900, 0.1860], [20, 0.9822, 0.2480], [25, 0.9730, 0.3100],
    [30, 0.9600, 0.3720], [35, 0.9427, 0.4340], [40, 0.9216, 0.4958],
    [45, 0.8962, 0.5571], [50, 0.8679, 0.6176], [55, 0.8350, 0.6769],
    [60, 0.7986, 0.7346], [65, 0.7597, 0.7903], [70, 0.7186, 0.8435],
    [75, 0.6732, 0.8936], [80, 0.6213, 0.9394], [85, 0.5722, 0.9761],
    [90, 0.5322, 1.0000],
])


def p_robinson(lon, lat):
    deg = np.abs(lat) / DEG
    xa = np.interp(deg, _ROB[:, 0], _ROB[:, 1])   # parallel length factor
    ya = np.interp(deg, _ROB[:, 0], _ROB[:, 2])   # distance from equator
    x = 0.8487 * xa * lon
    y = 1.3523 * ya * np.sign(lat)
    return x, y


def p_winkel(lon, lat):
    phi1 = math.acos(2.0 / math.pi)              # standard parallel
    a = np.arccos(np.clip(np.cos(lat) * np.cos(lon / 2.0), -1, 1))
    sinc = np.where(np.abs(a) < 1e-12, 1.0, np.sin(a) / np.where(a == 0, 1, a))
    x = 0.5 * (lon * math.cos(phi1) + 2 * np.cos(lat) * np.sin(lon / 2.0) / sinc)
    y = 0.5 * (lat + np.sin(lat) / sinc)
    return x, y


# --- cylindrical family (the "wall-map / wooden-map" look) ------------------
# All have straight horizontal parallels + vertical meridians. They differ only
# in how fast spacing grows toward the poles, i.e. how much they inflate the
# north. mercator > miller > gall > patterson (least polar inflation).

_MAXLAT = 85.0 * DEG   # safety clamp for conformal stretch

def p_mercator(lon, lat):
    lat = np.clip(lat, -_MAXLAT, _MAXLAT)
    return lon, np.arcsinh(np.tan(lat))

def p_miller(lon, lat):
    return lon, 1.25 * np.arcsinh(np.tan(0.8 * lat))

def p_gall(lon, lat):                             # Gall stereographic
    return lon / math.sqrt(2), (1 + math.sqrt(2) / 2) * np.tan(lat / 2)

def p_patterson(lon, lat):                        # Patterson cylindrical (2015)
    p2 = lat * lat
    y = lat * (1.0148 + p2 * p2 * (0.23185 + p2 * (-0.14499 + p2 * 0.02406)))
    return lon, y


PROJECTIONS = {
    'equalearth': p_equalearth,
    'mollweide':  p_mollweide,
    'robinson':   p_robinson,
    'winkel':     p_winkel,
    'mercator':   p_mercator,
    'miller':     p_miller,
    'gall':       p_gall,
    'patterson':  p_patterson,
}


# ---- coastline low-pass filter ---------------------------------------------

def smooth_ring(lon, lat, sigma_km):
    """Gaussian low-pass a closed lon/lat ring in arc-length (km) space.

    The ring is first resampled to uniform arc-length spacing (so unevenly
    spaced source vertices don't create wobble), then convolved with a
    circular Gaussian. Inlets/peninsulas whose along-coast extent is below
    ~ a few sigma are smoothed away. Returns the resampled, smoothed ring.
    """
    n = len(lon)
    if n < 8 or sigma_km <= 0:
        return lon, lat
    latr = lat * DEG
    # cumulative arc length (km), local equirectangular metric per segment
    dlon = np.diff(lon, append=lon[0])
    dlat = np.diff(lat, append=lat[0])
    midlat = (latr + np.roll(latr, -1)) / 2.0
    seg = np.sqrt((dlon * np.cos(midlat))**2 + dlat**2) * (DEG * R_EARTH_KM)
    total = seg.sum()
    if total <= 4 * sigma_km:          # ring smaller than the kernel: collapse-safe skip
        return lon, lat
    s = np.concatenate(([0.0], np.cumsum(seg)))      # length n+1, s[-1]=total

    # uniform resample (~4 samples per sigma). Cap is high so spacing stays
    # tied to sigma even on the giant continent rings (Norway lives on one).
    ds = sigma_km / 4.0
    m = int(min(max(total / ds, 16), 400000))
    su = np.linspace(0.0, total, m, endpoint=False)
    lon_c = np.concatenate((lon, lon[:1]))
    lat_c = np.concatenate((lat, lat[:1]))
    lon_u = np.interp(su, s, lon_c)
    lat_u = np.interp(su, s, lat_c)

    # circular Gaussian kernel in sample units
    dsr = total / m
    sig = sigma_km / dsr
    K = int(math.ceil(3 * sig))
    k = np.arange(-K, K + 1)
    w = np.exp(-0.5 * (k / sig)**2)
    w /= w.sum()
    out_lon = np.zeros(m)
    out_lat = np.zeros(m)
    for wj, kj in zip(w, k):
        out_lon += wj * np.roll(lon_u, -kj)
        out_lat += wj * np.roll(lat_u, -kj)
    # the smoothed curve is band-limited; decimate to ~2 samples/sigma so we
    # don't emit far more vertices than the original.
    stride = max(1, int(round(sig / 2.0)))   # sig is samples per sigma (~4) -> ~2/sigma out
    return out_lon[::stride], out_lat[::stride]


# ---- output -----------------------------------------------------------------

def fmt(v):
    return f'{v:.2f}'.rstrip('0').rstrip('.') or '0'


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', default='World_map_cleaned.svg')
    ap.add_argument('-o', '--output', help='output path (default World_map_<proj>.svg)')
    ap.add_argument('-p', '--projection', default='gall',
                    choices=list(PROJECTIONS))
    ap.add_argument('--width', type=float, default=WIDTH,
                    help='output canvas width in px (height derived)')
    ap.add_argument('--margin', type=float, default=20.0)
    ap.add_argument('--max-fjord-length', type=float, default=0.0,
                    help='coastline low-pass cutoff in km (0 = off). '
                         'Inlets/peninsulas shorter than ~this are smoothed.')
    args = ap.parse_args()

    out = args.output or f'World_map_{args.projection}.svg'
    sigma_km = args.max_fjord_length / math.pi   # cutoff wavelength -> Gaussian sigma

    svg = open(args.input, encoding='utf-8').read()
    dm = re.search(r'\bd="([^"]+)"', svg)
    if not dm:
        sys.exit('no <path d=...> found')
    subs = parse_subpaths(dm.group(1))

    proj = PROJECTIONS[args.projection]

    # project every subpath; collect for global bbox
    rings = []
    for s in subs:
        pts = np.asarray(s['pts'], dtype=float)
        lon = pts[:, 0] / WIDTH * 360.0 - 180.0
        lat = (LAT_B - pts[:, 1]) / LAT_M
        if sigma_km > 0:
            lon, lat = smooth_ring(lon, lat, sigma_km)
        X, Y = proj(lon * DEG, lat * DEG)
        rings.append((X, Y))

    allX = np.concatenate([r[0] for r in rings])
    allY = np.concatenate([r[1] for r in rings])
    minX, maxX, minY, maxY = allX.min(), allX.max(), allY.min(), allY.max()
    scale = (args.width - 2 * args.margin) / (maxX - minX)
    cw = args.width
    ch = (maxY - minY) * scale + 2 * args.margin

    def px(X): return (X - minX) * scale + args.margin
    def py(Y): return (maxY - Y) * scale + args.margin   # flip to SVG y-down

    parts = []
    for X, Y in rings:
        sx = px(X); sy = py(Y)
        seg = ['M', fmt(sx[0]), fmt(sy[0]), 'L']
        for j in range(1, len(sx)):
            seg.append(fmt(sx[j])); seg.append(fmt(sy[j]))
        seg.append('Z')
        parts.append(' '.join(seg))
    d = ''.join(parts)

    fill = re.search(r'fill="([^"]+)"', svg)
    fill = fill.group(1) if fill else '#bcbcbc'
    new = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{cw:.2f}" height="{ch:.2f}" '
           f'viewBox="0 0 {cw:.2f} {ch:.2f}" version="1.0">'
           f'<path fill="{fill}" fill-rule="nonzero" d="{d}"/></svg>')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(new)
    print(f'{args.projection}: {len(subs)} subpaths -> {out}  '
          f'({cw:.0f}x{ch:.0f}px)'
          + (f', fjord low-pass {args.max_fjord_length:.0f}km' if sigma_km > 0 else ''))


if __name__ == '__main__':
    main()
