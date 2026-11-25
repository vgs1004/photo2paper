"""PaperToys-style papercraft: a few cylinders and cones with the picture printed on.

Instead of folding the exact 3D shape out of ~150 triangles (see unfold.py),
this slices the AI model from top to bottom, fits a handful of round parts
(cylinders and cones) to its outline, and prints the model's outer surface on
them, arms and face included, the way hand-designed paper toys do. Each part
is a closed drum that stacks on the one below.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

import unfold as U

TEETH = 4.0          # mm, glue teeth on top/bottom rims
SIDE_TAB = 7.0       # mm, side glue strip
GRID_MM = 0.5        # resolution of the wrap-around picture
LETTERS = "ABCDEFGHIJKL"


# ------------------------------------------------------------------ profile
def _surface_samples(mesh, scale, sampler, n=700_000):
    pts, fidx = trimesh.sample.sample_surface(mesh, n)
    pts = pts * scale
    if sampler is not None:
        cols = sampler.sample(pts, mesh.face_normals[fidx]).astype(np.float32)
    else:
        cols = None
    return pts, cols


def _profile(pts, nslices=120, nang=48):
    """Radius of the model at each height: median over directions of the outermost point."""
    y = pts[:, 1]
    y0, y1 = y.min(), y.max()
    centre = np.median(pts[:, [0, 2]], axis=0)
    d = pts[:, [0, 2]] - centre
    r = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 0], d[:, 1])
    hb = np.clip(((y - y0) / (y1 - y0) * nslices).astype(int), 0, nslices - 1)
    ab = np.clip(((th + np.pi) / (2 * np.pi) * nang).astype(int), 0, nang - 1)
    rmax = np.zeros((nslices, nang))
    np.maximum.at(rmax, (hb, ab), r)
    prof = np.array([np.median(row[row > 0]) if (row > 0).any() else 0.0 for row in rmax])
    prof = ndimage.median_filter(prof, size=5, mode="nearest")
    hs = y0 + (np.arange(nslices) + 0.5) / nslices * (y1 - y0)
    return hs, prof, centre, (y0, y1)


def _fit_segments(hs, prof, max_parts, min_frac=0.12):
    """Split the profile into <= max_parts straight pieces (cones), least squares, DP."""
    n = len(hs)
    minlen = max(3, int(n * min_frac))
    # cost[i][j] = squared error of a straight line through prof[i:j]
    cost = np.full((n + 1, n + 1), np.inf)
    for i in range(n):
        for j in range(i + minlen, n + 1):
            x, yv = hs[i:j], prof[i:j]
            A = np.c_[x, np.ones_like(x)]
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            cost[i, j] = ((A @ coef - yv) ** 2).sum()
    best = None
    tol = 0.035 * prof.max()            # outline within ~3.5% of the widest radius is good enough
    for k in range(1, max_parts + 1):
        dp = np.full((k + 1, n + 1), np.inf)
        back = np.zeros((k + 1, n + 1), int)
        dp[0, 0] = 0
        for s in range(1, k + 1):
            for j in range(1, n + 1):
                cand = dp[s - 1, :j] + cost[:j, j]
                i = int(np.argmin(cand))
                dp[s, j], back[s, j] = cand[i], i
        err = dp[k, n]
        rms = math.sqrt(err / n)
        # stop adding parts once another one barely changes the outline
        # (fewer parts = easier to build)
        if best is not None and (best[2] < tol or rms > 0.85 * best[2]):
            break
        cuts, j = [], n
        for s in range(k, 0, -1):
            i = back[s, j]
            cuts.append((i, j))
            j = i
        best = (err, cuts[::-1], rms)
    segs = []
    for i, j in best[1]:
        x, yv = hs[i:j], prof[i:j]
        A = np.c_[x, np.ones_like(x)]
        (a, b), *_ = np.linalg.lstsq(A, yv, rcond=None)
        lo = hs[i] - (hs[1] - hs[0]) / 2
        hi = hs[j - 1] + (hs[1] - hs[0]) / 2
        segs.append([lo, hi, max(0.5, a * lo + b), max(0.5, a * hi + b)])
    return segs  # bottom-to-top: [y_bottom, y_top, r_bottom, r_top]


# ------------------------------------------------------------- picture grid
def _wrap_picture(pts, cols, centre, yrange, flat_colour):
    """Cylindrical 'outside view' picture: for each (height, angle) the outermost colour."""
    y0, y1 = yrange
    d = pts[:, [0, 2]] - centre
    r = np.hypot(d[:, 0], d[:, 1])
    H = max(8, int((y1 - y0) / GRID_MM))
    A = int(np.clip(2 * np.pi * np.percentile(r, 95) / GRID_MM, 256, 2048))
    img = np.zeros((H, A, 3), np.float32)
    if cols is None:
        img[:] = flat_colour
        return img
    th = np.arctan2(d[:, 0], d[:, 1])             # 0 = front (+Z), seam at the back
    hb = np.clip(((pts[:, 1] - y0) / (y1 - y0) * H).astype(int), 0, H - 1)
    ab = np.clip(((th + np.pi) / (2 * np.pi) * A).astype(int), 0, A - 1)
    order = np.argsort(r)                          # last write wins = outermost
    img[hb[order], ab[order]] = cols[order]
    rmap = np.zeros((H, A), np.float32)
    rmap[hb[order], ab[order]] = r[order]
    filled = rmap > 0
    # a bin that only saw the body while its neighbours saw an arm in front is a
    # gap in the arm, not a view of the body: drop it and fill from neighbours
    near_front = ndimage.maximum_filter(rmap, size=5, mode="wrap")
    filled &= rmap >= near_front - 2.5
    _, (ir, ic) = ndimage.distance_transform_edt(~filled, return_indices=True)
    out = img[ir, ic]
    return ndimage.median_filter(out, size=(3, 3, 1))


def _cap_picture(pts, cols, centre, ylo, yhi, top, flat_colour, res=1.0):
    """Top- or bottom-down view of the model between two heights."""
    sel = (pts[:, 1] >= ylo) & (pts[:, 1] <= yhi)
    if cols is None or sel.sum() < 50:
        return None
    p, c = pts[sel], cols[sel]
    d = p[:, [0, 2]] - centre
    R = np.abs(d).max() + 1
    n = int(2 * R / res) + 1
    gx = np.clip(((d[:, 0] + R) / res).astype(int), 0, n - 1)
    gz = np.clip(((d[:, 1] + R) / res).astype(int), 0, n - 1)
    order = np.argsort(p[:, 1] if top else -p[:, 1])
    img = np.zeros((n, n, 3), np.float32)
    img[gz[order], gx[order]] = c[order]
    filled = np.zeros((n, n), bool)
    filled[gz, gx] = True
    _, (ir, ic) = ndimage.distance_transform_edt(~filled, return_indices=True)
    return img[ir, ic], R, res


# ------------------------------------------------------------------ pieces
def _sector_pieces(seg, letter, name):
    """Unroll one frustum into 1+ flat strips (annular sectors or rectangles)."""
    yb, yt, rb, rt = seg
    h = yt - yb
    s = math.hypot(h, rb - rt)
    lim_w = U.USABLE_W - 2 * SIDE_TAB
    lim_h = U.USABLE_H - 2 * TEETH - 4
    pieces = []
    cyl = abs(rb - rt) < 0.03 * s
    if cyl:
        rm = (rb + rt) / 2
        circ = 2 * math.pi * rm
        n = max(1, math.ceil(circ / lim_w))
        w = circ / n
        for k in range(n):
            a0, a1 = -math.pi + 2 * math.pi * k / n, -math.pi + 2 * math.pi * (k + 1) / n
            outline = [(0, 0), (w, 0), (w, s), (0, s)]
            pieces.append(dict(kind="rect", w=w, s=s, a0=a0, a1=a1, seg=seg,
                               outline=outline, bottom=[(0, 0), (w, 0)], top=[(0, s), (w, s)],
                               left=[(0, 0), (0, s)], right=[(w, 0), (w, s)]))
    else:
        big, small = max(rb, rt), min(rb, rt)
        R1 = big * s / (big - small)               # apex to the wide rim
        R2 = R1 - s
        phi = 2 * math.pi * big / R1
        n = 1
        while True:
            dphi = phi / n
            if _sector_fits(R1, R2, dphi, lim_w, lim_h):
                break
            n += 1
            if n > 12:
                break
        for k in range(n):
            a0, a1 = -math.pi + 2 * math.pi * k / n, -math.pi + 2 * math.pi * (k + 1) / n
            b0, b1 = -phi / 2 + phi * k / n, -phi / 2 + phi * (k + 1) / n
            arc = lambda rho: [(rho * math.sin(b), -rho * math.cos(b)) for b in np.linspace(b0, b1, 48)]
            outer, inner = arc(R1), arc(max(R2, 0.0))
            outline = outer + inner[::-1]
            wide_is_bottom = rb >= rt
            pieces.append(dict(kind="sector", R1=R1, R2=R2, b0=b0, b1=b1, a0=a0, a1=a1, seg=seg,
                               wide_bottom=wide_is_bottom, outline=outline,
                               bottom=outer if wide_is_bottom else inner,
                               top=inner if wide_is_bottom else outer,
                               left=[inner[0], outer[0]], right=[inner[-1], outer[-1]]))
    for i, p in enumerate(pieces):
        p["label"] = f"{letter}{i + 1}" if len(pieces) > 1 else letter
        p["part"], p["name"], p["count"], p["index"] = letter, name, len(pieces), i
    return pieces


def _sector_fits(R1, R2, dphi, lim_w, lim_h):
    pts = [(rho * math.sin(b), -rho * math.cos(b)) for rho in (R1, R2) for b in np.linspace(-dphi / 2, dphi / 2, 40)]
    poly = Polygon(pts).convex_hull
    rect = poly.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    e = sorted(math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(2))
    return e[0] <= min(lim_w, lim_h) and e[1] <= max(lim_w, lim_h)


def _teeth(line, outward_ref, depth=TEETH):
    """Small trapezoid glue teeth along a polyline, pointing away from outward_ref."""
    pts = np.asarray(line, float)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    L = seglen[-1]
    if L < 6:
        return []
    n = max(2, int(L / 9))
    teeth = []
    ref = np.asarray(outward_ref, float)
    for k in range(n):
        s0, s1 = L * (k + 0.15) / n, L * (k + 0.85) / n
        p0 = np.array([np.interp(s0, seglen, pts[:, 0]), np.interp(s0, seglen, pts[:, 1])])
        p1 = np.array([np.interp(s1, seglen, pts[:, 0]), np.interp(s1, seglen, pts[:, 1])])
        d = (p1 - p0) / max(1e-9, np.linalg.norm(p1 - p0))
        nrm = np.array([-d[1], d[0]])
        if np.dot((p0 + p1) / 2 - ref, nrm) < 0:
            nrm = -nrm
        inset = min(depth * 0.6, np.linalg.norm(p1 - p0) * 0.3)
        teeth.append(Polygon([p0, p1, p1 + nrm * depth - d * inset, p0 + nrm * depth + d * inset]))
    return teeth


def _side_tab(line, ref, width=SIDE_TAB):
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    d = (b - a) / np.linalg.norm(b - a)
    nrm = np.array([-d[1], d[0]])
    if np.dot((a + b) / 2 - np.asarray(ref), nrm) < 0:
        nrm = -nrm
    inset = min(width * 0.7, np.linalg.norm(b - a) * 0.2)
    return Polygon([a, b, b + nrm * width - d * inset, a + nrm * width + d * inset])


def _finish_piece(p, has_bottom_cap, has_top_cap):
    body = Polygon(p["outline"]).buffer(0)
    ref = np.array(body.centroid.coords[0])
    tabs = []
    if p["count"] > 1 or True:
        tabs.append(("side", _side_tab(p["right"], ref)))
    if has_bottom_cap:
        tabs += [("teeth", t) for t in _teeth(p["bottom"], ref)]
    if has_top_cap:
        tabs += [("teeth", t) for t in _teeth(p["top"], ref)]
    p["body"] = body
    p["tabs"] = tabs
    p["outline_full"] = unary_union([body] + [t for _, t in tabs])


def _disc_piece(radius, letter, which, pic):
    circle = [(radius * math.cos(a), radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 96, endpoint=False)]
    body = Polygon(circle)
    return dict(kind="disc", r=radius, label=f"{letter} {which}".strip(), part=letter, which=which or "top",
                body=body, tabs=[], outline_full=body, pic=pic)


# ------------------------------------------------------------------ render
def _piece_image(p, wrap, yrange, flat_colour):
    from PIL import Image
    minx, miny, maxx, maxy = p["body"].buffer(1.0).bounds
    W = int(math.ceil((maxx - minx) * U.PX_PER_MM))
    H = int(math.ceil((maxy - miny) * U.PX_PER_MM))
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    x = minx + (cols + 0.5) / U.PX_PER_MM
    y = maxy - (rows + 0.5) / U.PX_PER_MM
    y0, y1 = yrange
    GH, GA = wrap.shape[:2]
    if p["kind"] == "disc":
        rgb = np.zeros((H, W, 3), np.float32)
        rgb[:] = flat_colour
        if p["pic"] is not None:
            img, R, res = p["pic"]
            gx = np.clip(((x + R) / res).astype(int), 0, img.shape[1] - 1)
            gz = np.clip(((-y if p["which"] == "top" else y) + R) / res, 0, img.shape[0] - 1).astype(int)
            rgb = img[gz, gx]
        inside = np.hypot(x, y) <= p["r"] + 1.0
    else:
        yb, yt, rb, rt = p["seg"]
        if p["kind"] == "rect":
            frac_a = np.clip(x / p["w"], 0, 1)
            frac_h = np.clip(y / p["s"], 0, 1)
            inside = (x >= -1) & (x <= p["w"] + 1) & (y >= -1) & (y <= p["s"] + 1)
        else:
            rho = np.hypot(x, y)
            b = np.arctan2(x, -y)
            frac_a = np.clip((b - p["b0"]) / (p["b1"] - p["b0"]), 0, 1)
            R1, R2 = p["R1"], p["R2"]
            t = np.clip((R1 - rho) / (R1 - R2), 0, 1)   # 0 at wide rim, 1 at narrow rim
            frac_h = t if p["wide_bottom"] else 1 - t
            inside = (rho <= R1 + 1) & (rho >= R2 - 1) & (b >= p["b0"] - 0.02) & (b <= p["b1"] + 0.02)
        theta = p["a0"] + frac_a * (p["a1"] - p["a0"])
        hh = yb + frac_h * (yt - yb)
        gi = np.clip(((hh - y0) / (y1 - y0) * GH).astype(int), 0, GH - 1)
        gj = np.clip(((theta + np.pi) / (2 * np.pi) * GA).astype(int), 0, GA - 1)
        rgb = wrap[gi, gj]
    alpha = (inside * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha]), "RGBA")
    return img, minx, miny, maxx - minx, maxy - miny


def _draw(c, poly, T, fill=None, stroke=(0, 0, 0), width=0.6, dash=None):
    from reportlab.lib.colors import Color
    pts = [T(q) for q in list(poly.exterior.coords)[:-1]]
    path = c.beginPath()
    path.moveTo(*pts[0])
    for q in pts[1:]:
        path.lineTo(*q)
    path.close()
    if fill is not None:
        c.setFillColor(Color(*fill))
    c.setStrokeColor(Color(*stroke))
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.drawPath(path, fill=1 if fill is not None else 0, stroke=1)
    c.setDash([])


def simple_papercraft(mesh_path, out_pdf, size_mm=150.0, title="Papercraft", preview_png=None,
                      paper="A4", max_parts=4, log=print):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import Color
    from reportlab.lib.utils import ImageReader

    U.set_paper(paper)
    mesh = U.load_mesh(mesh_path)
    scale = size_mm / mesh.extents.max()
    flat = U.photo_colour(preview_png) if preview_png else np.array([200.0, 200.0, 200.0])
    sampler = U.TextureSampler(mesh, scale) if U.usable_texture(mesh) else None
    log("printing the model's picture" if sampler else f"no usable texture; using colour {flat.round().astype(int).tolist()}")
    pts, cols = _surface_samples(mesh, scale, sampler)
    hs, prof, centre, yrange = _profile(pts)
    segs = _fit_segments(hs, prof, max_parts)
    # parts taller than a sheet get split in two
    split = []
    for s in segs:
        yb, yt, rb, rt = s
        n = max(1, math.ceil(math.hypot(yt - yb, rb - rt) / (U.USABLE_H - 2 * TEETH - 10)))
        for k in range(n):
            f0, f1 = k / n, (k + 1) / n
            split.append([yb + f0 * (yt - yb), yb + f1 * (yt - yb), rb + f0 * (rt - rb), rb + f1 * (rt - rb)])
    segs = split[::-1]                                  # top-to-bottom for lettering
    log(f"fitted {len(segs)} round parts: " + ", ".join(
        f"{'cylinder' if abs(s[2]-s[3]) < 0.03*abs(s[1]-s[0]) + 0.5 else 'cone'} {2*max(s[2],s[3]):.0f} mm wide" for s in segs))
    wrap = _wrap_picture(pts, cols, centre, yrange, flat)

    names = {0: "top"}
    pieces = []
    for i, seg in enumerate(segs):
        letter = LETTERS[i]
        yb, yt, rb, rt = seg
        tip_top = rt < 2.5
        tip_bottom = rb < 2.5
        strips = _sector_pieces(seg, letter, names.get(i, ""))
        for p in strips:
            _finish_piece(p, not tip_bottom, not tip_top)
        pieces += strips
        span = (yt - yb) * 0.5
        # one disc closes the top of the model; one disc per joint is shared by the
        # part above and the part below; one disc closes the bottom
        if i == 0 and not tip_top:
            pieces.append(_disc_piece(rt, letter, "top", _cap_picture(pts, cols, centre, yt - span, np.inf, True, flat)))
        if i + 1 < len(segs):
            below = LETTERS[i + 1]
            r_joint = max(rb, segs[i + 1][3])
            ylo = segs[i + 1][0] + (segs[i + 1][1] - segs[i + 1][0]) * 0.5
            pieces.append(_disc_piece(r_joint, f"{letter}/{below}", "", _cap_picture(pts, cols, centre, ylo, yb, True, flat)))
        elif not tip_bottom:
            pieces.append(_disc_piece(rb, letter, "bottom", _cap_picture(pts, cols, centre, -np.inf, yb + span, False, flat)))

    pages = U.layout(pieces)
    c = canvas.Canvas(out_pdf, pagesize=(U.PAGE_W * U.MM, U.PAGE_H * U.MM))
    c.setTitle(title)
    MM = U.MM

    # ---------- instructions page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(U.MARGIN * MM, (U.PAGE_H - 20) * MM, title)
    c.setFont("Helvetica", 10)
    lines = [
        f"Simple-shapes model, about {size_mm:.0f} mm tall/long. {len(segs)} parts, "
        f"{len(pages)} {U.paper_name()} sheet(s). Print at 100% on 160-200 gsm card.",
        "",
        "1. Cut out every piece along the solid line, including the glue tabs.",
        "2. Fold all tabs and teeth backwards (dashed lines).",
        "3. Roll each lettered piece into a tube or cone, picture outwards, and glue the side",
        "   tab under the opposite edge. Pieces like B1 + B2 join into one ring first.",
        "4. Build from the bottom up. Each disc labelled with two letters (\"B/C\") joins those",
        "   two parts: glue the teeth of the lower part to the underside of the disc and the",
        "   teeth of the upper part to its top. The wider part's teeth go under the disc.",
        "5. Close the very top and bottom with the discs marked \"top\" and \"bottom\".",
    ]
    yy = U.PAGE_H - 32
    for ln in lines:
        c.drawString(U.MARGIN * MM, yy * MM, ln)
        yy -= 5.2
    # side view of the stack, coloured from the picture's front column
    yy -= 6
    total_h = sum(s[1] - s[0] for s in segs)
    maxr = max(max(s[2], s[3]) for s in segs)
    sc = min(60 / total_h, 40 / maxr)
    cx = U.MARGIN + 50
    ytop = yy
    for i, (yb, yt, rb, rt) in enumerate(segs):
        y_hi = ytop - (yrange[1] - yt) * sc
        y_lo = ytop - (yrange[1] - yb) * sc
        mid = (yb + yt) / 2
        gi = int(np.clip((mid - yrange[0]) / (yrange[1] - yrange[0]) * wrap.shape[0], 0, wrap.shape[0] - 1))
        col = wrap[gi, wrap.shape[1] // 2] / 255.0
        p = c.beginPath()
        p.moveTo((cx - rb * sc) * MM, y_lo * MM)
        p.lineTo((cx + rb * sc) * MM, y_lo * MM)
        p.lineTo((cx + rt * sc) * MM, y_hi * MM)
        p.lineTo((cx - rt * sc) * MM, y_hi * MM)
        p.close()
        c.setFillColor(Color(*col))
        c.setStrokeColor(Color(0, 0, 0))
        c.setLineWidth(0.6)
        c.drawPath(p, fill=1, stroke=1)
        c.setFillColor(Color(0, 0, 0))
        c.setFont("Helvetica-Bold", 9)
        c.drawString((cx + maxr * sc + 6) * MM, ((y_lo + y_hi) / 2 - 1) * MM, LETTERS[i])
    c.setFont("Helvetica", 8)
    c.drawString((cx - 30) * MM, (ytop - total_h * sc - 6) * MM, "How the parts stack (side view)")
    if preview_png:
        try:
            img = ImageReader(preview_png)
            iw, ih = img.getSize()
            s = min(70 / iw, 70 / ih)
            c.drawImage(img, (U.PAGE_W - U.MARGIN - iw * s) * MM, (ytop - ih * s) * MM,
                        iw * s * MM, ih * s * MM, mask="auto")
        except Exception:
            pass
    c.showPage()

    # ---------- sheets
    for pno, page in enumerate(pages, 1):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(U.MARGIN * MM, (U.PAGE_H - U.MARGIN - 4) * MM, f"{title}  -  sheet {pno}/{len(pages)}")
        for pl in page:
            p = pl["piece"]
            ang = pl["angle"]
            th = math.radians(ang)
            cs, sn = math.cos(th), math.sin(th)
            dx, dy = pl["off"]
            ax, ay = pl["at"]
            ay = U.USABLE_H - ay - pl["h"]
            ox = dx + ax + U.MARGIN
            oy = dy + ay + U.MARGIN

            def T(q, cs=cs, sn=sn, ox=ox, oy=oy):
                x, y = q
                return (x * cs - y * sn + ox) * MM, (x * sn + y * cs + oy) * MM

            for kind, tab in p["tabs"]:
                _draw(c, tab, T, fill=(0.88, 0.88, 0.88), width=0.5)
            img, mx, my, w, h = _piece_image(p, wrap, yrange, flat)
            c.saveState()
            c.translate(ox * MM, oy * MM)
            c.rotate(ang)
            c.drawImage(ImageReader(img), mx * MM, my * MM, w * MM, h * MM, mask="auto")
            c.restoreState()
            _draw(c, p["body"], T, width=0.6)
            # fold lines where tabs meet the body
            c.setLineWidth(0.35)
            c.setStrokeColor(Color(0.2, 0.2, 0.2))
            c.setDash([3, 2])
            for kind, tab in p["tabs"]:
                q = list(tab.exterior.coords)
                c.line(*T(q[0]), *T(q[1]))
            c.setDash([])
            # label
            cen = p["body"].representative_point()
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(Color(1, 1, 1, alpha=0.75))
            x, y = T((cen.x, cen.y))
            lw = c.stringWidth(p["label"], "Helvetica-Bold", 9) + 8
            c.roundRect(x - lw / 2, y - 4, lw, 12, 3, stroke=0, fill=1)
            c.setFillColor(Color(0, 0, 0))
            c.drawCentredString(x, y, p["label"])
            for kind, tab in p["tabs"]:
                if kind == "side":
                    tc = tab.centroid
                    c.setFont("Helvetica", 5)
                    c.setFillColor(Color(0.3, 0.3, 0.3))
                    tx, ty = T((tc.x, tc.y))
                    c.drawCentredString(tx, ty - 1.5, "glue")
        c.showPage()
    c.save()
    log(f"wrote {out_pdf}: {len(pieces)} pieces on {len(pages)} {paper} sheet(s)")
    return {"parts": len(segs), "pieces": len(pieces), "sheets": len(pages)}
