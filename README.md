DONE:

- intact Greenland + South America copies are dropped into open ocean (within a
  single piece, clear of land) so a seam-free version can be cut and glued over
  the seamed original. Copies that fit no piece's ocean go on patches.svg.
  (worldmap_tiles.py, on by default; --no-copies to disable)
  South America is split off the connected Americas at its THINNEST neck (the
  Panama isthmus): erode until N/S pinch apart, then cut straight across the
  neck.
- vertical seam cut line drawn where each piece meets a neighbour, so the laser
  cuts each mating edge to the exact seam -- but GAPPED wherever land overhangs
  the seam (kept-whole islands), so intentional overhangs aren't sliced off.
  (--no-seam-edges to disable)
