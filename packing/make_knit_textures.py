"""Generate tileable knit textures (stockinette and 1x1 rib) for the folded sweater.

Poly Haven has no knit-stitch scan, so the stitch is modelled here: every stitch is a
"V" of two twisted yarn legs. Outputs diff/nor_gl/rough maps in the same layout as the
Poly Haven folders, into assets/textures/knit_stockinette_2k and knit_rib_2k.
One tile covers 8 cm (stockinette: 16 stitches x 20 rows, rib: 20 ribs x 20 rows).
Usage: python packing/make_knit_textures.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "assets" / "textures"
N = 2048
rng = np.random.default_rng(7)


def tile_noise(n, cells, octaves=4):
    """Periodic value noise in [0, 1] built from a random lattice with cosine smoothing."""
    acc = np.zeros((n, n)); amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = cells * 2 ** o
        g = rng.random((c, c))
        t = np.arange(n) * c / n
        i0 = np.floor(t).astype(int); f = t - i0; f = (1 - np.cos(np.pi * f)) / 2
        i1 = (i0 + 1) % c
        a = g[np.ix_(i0, i0)]; b = g[np.ix_(i0, i1)]; cc = g[np.ix_(i1, i0)]; d = g[np.ix_(i1, i1)]
        fy, fx = f[:, None], f[None, :]
        acc += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + cc * (1 - fx) * fy + d * fx * fy)
        tot += amp; amp *= 0.5
    return acc / tot


def stitch_field(cols, rows, rib=False):
    y, x = np.mgrid[0:N, 0:N] / N
    u = x * cols; v = (1 - y) * rows                      # v up, so the V opens upwards
    cu0 = np.floor(u); cv0 = np.floor(v)
    jit = rng.random((rows, cols, 2))
    h = np.zeros((N, N)); jv = np.zeros((N, N))
    tilt = np.radians(24)
    # each stitch: two rounded yarn legs (tilted ellipses) leaning outwards, overlapping
    # the row above; evaluate neighbouring cells too so the overlaps interlock
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            cu = cu0 + du; cv = cv0 + dv
            iu = cu.astype(int) % cols; iv = cv.astype(int) % rows
            j = jit[iv, iu]
            if rib:
                purl = (iu % 2 == 1)
            for side in (-1, 1):
                cx = cu + 0.5 + side * 0.23; cy = cv + 0.55 + 0.04 * (j[..., 1] - 0.5)
                dx = u - cx; dy = v - cy
                a = side * tilt
                rx = dx * np.cos(a) - dy * np.sin(a); ry = dx * np.sin(a) + dy * np.cos(a)
                r2 = (rx / 0.27) ** 2 + (ry / 0.64) ** 2
                leg = np.sqrt(np.clip(1 - r2, 0, 1))
                # yarn twist: fine ridges across the leg
                twist = 0.5 + 0.5 * np.cos((ry * 6.5 + rx * 2.0) * 2 * np.pi)
                leg = leg * (0.90 + 0.10 * twist) * (0.9 + 0.2 * j[..., 0])
                # the leg ends tuck under the next row: lower top part slightly
                leg = leg * (1 - 0.25 * np.clip(ry / 0.6, 0, 1))
                if rib:
                    leg = np.where(purl, 0.0, leg)
                take = leg > h
                h = np.where(take, leg, h); jv = np.where(take, j[..., 0], jv)
    if rib:
        # purl columns: small horizontal loops deep in the fabric
        fu = u - cu0 - 0.5; fv = v - cv0
        purl = (cu0.astype(int) % cols % 2 == 1)
        loop = np.sqrt(np.clip(1 - (fu / 0.5) ** 2 - ((fv - 0.5) / 0.3) ** 2, 0, 1)) * 0.35
        h = np.where(purl, np.maximum(h, loop - 0.15), h + 0.1)
    fuzz = tile_noise(N, 64, 3)
    h = h + 0.04 * (fuzz - 0.5)
    return h, jv, fuzz


def write(name, h, jv, fuzz, texel_m):
    folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
    # normal map (OpenGL convention: +Y up), height in units of yarn diameter ~ 1.2 mm
    height_m = h * 0.0012
    gy, gx = np.gradient(height_m, texel_m)
    # periodic fix at the tile border is not needed: the field is periodic by construction
    nx, ny = -gx, gy
    nz = np.ones_like(nx)
    ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nor = np.stack([nx / ln, ny / ln, nz / ln], -1) * 0.5 + 0.5
    Image.fromarray((nor * 255).astype(np.uint8)).save(folder / "nor_gl.jpg", quality=94)
    # albedo: near-white yarn (tinted in the shader); crevices darker, slight melange
    hn = (h - h.min()) / (h.max() - h.min())
    ao = 0.55 + 0.45 * hn ** 0.7
    mel = 0.94 + 0.06 * jv + 0.05 * (fuzz - 0.5)
    alb = np.clip(ao * mel, 0, 1)
    rgb = np.stack([alb, alb * 0.995, alb * 0.985], -1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(folder / "diff.jpg", quality=94)
    rough = np.clip(0.80 + 0.12 * (1 - hn) + 0.05 * (fuzz - 0.5), 0, 1)
    Image.fromarray((rough * 255).astype(np.uint8)).save(folder / "rough.jpg", quality=94)
    print("wrote", folder)


def main():
    texel = 0.08 / N
    h, jv, fz = stitch_field(16, 20)
    write("knit_stockinette_2k", h, jv, fz, texel)
    h, jv, fz = stitch_field(20, 20, rib=True)
    write("knit_rib_2k", h, jv, fz, texel)


if __name__ == "__main__":
    main()
