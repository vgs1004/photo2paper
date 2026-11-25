"""Mesh -> printable papercraft PDF.

Takes any triangle mesh (GLB / OBJ / PLY / STL), simplifies it to a
paper-friendly number of faces, unfolds it into flat pieces that do not
overlap, adds numbered glue tabs, and lays the pieces out on A4 pages.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely import affinity

# ---------------------------------------------------------------- page setup
MM = 72.0 / 25.4                     # PDF points per millimetre
PAGE_W, PAGE_H = 210.0, 297.0        # A4, mm
MARGIN = 10.0
HEADER = 14.0                        # space for the page title
USABLE_W = PAGE_W - 2 * MARGIN
USABLE_H = PAGE_H - 2 * MARGIN - HEADER

PAPERS = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "Letter": (215.9, 279.4)}


def set_paper(name: str):
    """Switch the page size used by unfolding, layout and rendering."""
    global PAGE_W, PAGE_H, USABLE_W, USABLE_H
    if name not in PAPERS:
        raise ValueError(f"paper must be one of {', '.join(PAPERS)}")
    PAGE_W, PAGE_H = PAPERS[name]
    USABLE_W = PAGE_W - 2 * MARGIN
    USABLE_H = PAGE_H - 2 * MARGIN - HEADER
GAP = 4.0                            # space between pieces
TAB_H = 5.0                          # glue tab height, mm


# ------------------------------------------------------------ mesh handling
def load_mesh(path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(path, force="scene")
    meshes = []
    for name, geom in scene_or_mesh.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        g = geom.copy()
        try:
            transform = scene_or_mesh.graph.get(name)[0]
            g.apply_transform(transform)
        except Exception:
            pass
        meshes.append(g)
    if not meshes:
        raise ValueError(f"No triangle mesh found in {path}")
    mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    return mesh


def face_colors_of(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-face RGB (0-255) sampled from texture / vertex colours."""
    vis = mesh.visual
    try:
        if vis.kind == "texture" and vis.uv is not None:
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                tex = np.asarray(img.convert("RGB"), dtype=float)
                h, w = tex.shape[:2]
                # sample the texture at each face's three corners + centre
                uv = np.asarray(vis.uv, dtype=float)[mesh.faces]          # (F,3,2)
                uv = np.concatenate([uv, uv.mean(axis=1, keepdims=True)], axis=1)
                u = np.clip(uv[..., 0] % 1.0, 0, 1) * (w - 1)
                v = (1 - np.clip(uv[..., 1] % 1.0, 0, 1)) * (h - 1)
                fc = tex[v.round().astype(int), u.round().astype(int)].mean(axis=1)
                factor = getattr(mat, "baseColorFactor", None)
                if factor is not None:
                    fc = fc * (np.asarray(factor[:3], dtype=float) / 255.0
                               if np.max(factor) > 1 else np.asarray(factor[:3]))
                return fc
    except Exception:
        pass
    try:
        if vis.kind == "texture":
            vis = vis.to_color()
        fc = np.asarray(vis.face_colors)[:, :3].astype(float)
        if fc.shape[0] == len(mesh.faces):
            return fc
    except Exception:
        pass
    return np.full((len(mesh.faces), 3), 235.0)


def _clean(v, f):
    """Weld, then drop folded-over duplicate faces (zero-volume fins) and slivers."""
    t = trimesh.Trimesh(v, f, process=True)
    groups = trimesh.grouping.group_rows(np.sort(t.faces, axis=1), require_count=None)
    keep = np.ones(len(t.faces), bool)
    for g in groups:
        if len(g) > 1:
            keep[g] = False
    keep &= t.area_faces > 1e-12
    t.update_faces(keep)
    t.remove_unreferenced_vertices()
    return t


def _decimate(mesh, target):
    """Simplify in stages; one huge jump tends to fold the surface over itself."""
    import fast_simplification
    v, f = mesh.vertices, mesh.faces
    stages = [s for s in (50000, 8000, 1500) if s > 3 * target and s < len(f)] + [target]
    t = None
    for s in stages:
        v, f = fast_simplification.simplify(
            np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64),
            target_reduction=max(0.0, 1.0 - s / len(f)))
        t = _clean(v, f)
        v, f = t.vertices, t.faces
    return t


def simplify(mesh: trimesh.Trimesh, target_faces: int):
    """Decimate and carry colours over from the detailed mesh."""
    colors = face_colors_of(mesh)
    centroids = mesh.triangles_center
    # geometry only: AI meshes split vertices along texture seams, which
    # leaves holes and blocks decimation, so weld by position alone
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:                      # drop floating specks
        big = max(len(p.faces) for p in parts)
        mesh = trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.05 * big])
    if len(mesh.faces) > target_faces:
        # try a few nearby targets and keep the first closed (watertight) result
        best = None
        for t in (target_faces, int(target_faces * 0.9), int(target_faces * 1.1),
                  int(target_faces * 0.8), int(target_faces * 1.2)):
            cand = _decimate(mesh, t)
            if best is None or (cand.is_watertight and not best.is_watertight):
                best = cand
            if cand.is_watertight:
                break
        simple = best
    else:
        simple = _clean(mesh.vertices, mesh.faces)
    trimesh.repair.fix_normals(simple)

    # colour of each simple face = average of nearby detailed faces
    tree = cKDTree(centroids)
    k = min(8, len(centroids))
    _, idx = tree.query(simple.triangles_center, k=k)
    idx = np.atleast_2d(idx).reshape(len(simple.faces), -1)
    simple_colors = colors[idx].mean(axis=1)
    return simple, simple_colors


# -------------------------------------------------------------- unfolding
@dataclass
class Island:
    faces: list = field(default_factory=list)
    pos: dict = field(default_factory=dict)       # face -> {vertex: (x, y)}
    polys: list = field(default_factory=list)
    union: Polygon | None = None


def _place_child(parent_pos, a, b, c, V):
    """2D position of vertex c of a face that hinges on edge a-b."""
    pa, pb = np.array(parent_pos[a]), np.array(parent_pos[b])
    A, B, C = V[a], V[b], V[c]
    ab = B - A
    L = np.linalg.norm(ab)
    t = np.dot(C - A, ab) / (L * L)
    foot = A + t * ab
    h = np.linalg.norm(C - foot)
    d2 = (pb - pa) / np.linalg.norm(pb - pa)
    perp = np.array([-d2[1], d2[0]])
    # parent's third vertex must be on the opposite side
    other = [v for v in parent_pos if v not in (a, b)][0]
    po = np.array(parent_pos[other])
    if np.dot(po - pa, perp) > 0:
        perp = -perp
    pc = pa + t * (pb - pa) + h * perp
    return {a: tuple(pa), b: tuple(pb), c: tuple(pc)}


def _first_face_pos(face, V):
    a, b, c = face
    A, B, C = V[a], V[b], V[c]
    e1 = B - A
    L = np.linalg.norm(e1)
    x = np.dot(C - A, e1) / L
    y = np.linalg.norm(np.cross(e1, C - A)) / L
    return {a: (0.0, 0.0), b: (L, 0.0), c: (x, y)}


def _fits_on_page(geom) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    lim_w, lim_h = USABLE_W - 2 * TAB_H, USABLE_H - 2 * TAB_H
    if (w <= lim_w and h <= lim_h) or (w <= lim_h and h <= lim_w):
        return True
    rect = geom.minimum_rotated_rectangle
    xs, ys = rect.exterior.coords.xy
    e = [math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(2)]
    w, h = sorted(e)
    return w <= min(lim_w, lim_h) and h <= max(lim_w, lim_h)


def unfold(mesh: trimesh.Trimesh):
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces)
    adj = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    angles = mesh.face_adjacency_angles
    convex = mesh.face_adjacency_convex

    neighbours = {i: [] for i in range(len(F))}
    edge_info = {}
    for k, ((f1, f2), (v1, v2)) in enumerate(zip(adj, adj_edges)):
        key = frozenset((f1, f2))
        edge_info[key] = (int(v1), int(v2), float(angles[k]), bool(convex[k]))
        neighbours[f1].append((f2, int(v1), int(v2), float(angles[k])))
        neighbours[f2].append((f1, int(v1), int(v2), float(angles[k])))

    areas = mesh.area_faces
    unplaced = set(range(len(F)))
    islands: list[Island] = []

    while unplaced:
        seed = max(unplaced, key=lambda f: areas[f])
        isl = Island()
        pos = _first_face_pos(F[seed], V)
        isl.faces.append(seed)
        isl.pos[seed] = pos
        poly = Polygon([pos[v] for v in F[seed]])
        isl.polys.append(poly)
        isl.union = poly
        unplaced.discard(seed)

        heap = []
        def push(f):
            for g, a, b, ang in neighbours[f]:
                if g in unplaced:
                    heapq.heappush(heap, (ang, f, g, a, b))
        push(seed)

        while heap:
            ang, f, g, a, b = heapq.heappop(heap)
            if g not in unplaced:
                continue
            c = [v for v in F[g] if v not in (a, b)][0]
            gpos = _place_child(isl.pos[f], a, b, c, V)
            gpoly = Polygon([gpos[v] for v in F[g]])
            if not gpoly.is_valid or gpoly.area < 1e-9:
                continue
            if isl.union.intersection(gpoly.buffer(-1e-3 * math.sqrt(gpoly.area))).area > 1e-6 * gpoly.area:
                continue
            new_union = isl.union.union(gpoly)
            if not _fits_on_page(new_union):
                continue
            isl.faces.append(g)
            isl.pos[g] = gpos
            isl.polys.append(gpoly)
            isl.union = new_union
            unplaced.discard(g)
            push(g)
        islands.append(isl)
    return islands, edge_info


# ---------------------------------------------------------- annotations
def _edge_key(a, b):
    return (min(a, b), max(a, b))


def annotate(islands, edge_info, mesh):
    """Classify every edge of every island as fold, cut+tab, or cut+number."""
    F = np.asarray(mesh.faces)
    face_island = {}
    for i, isl in enumerate(islands):
        for f in isl.faces:
            face_island[f] = i

    # which face pairs got joined inside an island (hinged)?
    joined = set()
    for isl in islands:
        placed = set(isl.faces)
        for f in isl.faces:
            for g in placed:
                if g <= f:
                    continue
                key = frozenset((f, g))
                if key not in edge_info:
                    continue
                a, b, _, _ = edge_info[key]
                if (np.allclose(isl.pos[f][a], isl.pos[g][a], atol=1e-6)
                        and np.allclose(isl.pos[f][b], isl.pos[g][b], atol=1e-6)):
                    joined.add(key)

    edge_faces = {}
    for key, (a, b, _, _) in edge_info.items():
        edge_faces[_edge_key(a, b)] = tuple(key)

    numbers = {}
    next_no = 1
    pieces = []
    for i, isl in enumerate(islands):
        piece = {"faces": [], "folds": [], "cuts": [], "tabs": [], "labels": []}
        for f in isl.faces:
            pts = [isl.pos[f][v] for v in F[f]]
            piece["faces"].append((f, pts))
            for j in range(3):
                a, b = int(F[f][j]), int(F[f][(j + 1) % 3])
                pa, pb = isl.pos[f][a], isl.pos[f][b]
                ek = _edge_key(a, b)
                pair = edge_faces.get(ek)
                if pair is None:                       # open border of mesh
                    piece["cuts"].append((pa, pb))
                    continue
                key = frozenset(pair)
                if key in joined:
                    if f == min(pair):                 # draw once
                        _, _, ang, cvx = edge_info[key]
                        if ang > math.radians(1.0):
                            piece["folds"].append((pa, pb, "mountain" if cvx else "valley"))
                    continue
                piece["cuts"].append((pa, pb))
                if ek not in numbers:
                    numbers[ek] = next_no
                    next_no += 1
                other = pair[0] if pair[1] == f else pair[1]
                piece["labels"].append((pa, pb, numbers[ek], f, other))
        pieces.append(piece)

    # decide tab side: try the lower-numbered face first, fall back if overlapping
    tab_owner = {}
    for pi, piece in enumerate(pieces):
        piece["outline"] = islands[pi].union
    for pi, piece in enumerate(pieces):
        for (pa, pb, no, f, other) in piece["labels"]:
            if no in tab_owner:
                continue
            first = f if f < other else other
            tab_owner[no] = first
    occupied = {pi: pieces[pi]["outline"] for pi in range(len(pieces))}
    for pi, piece in enumerate(pieces):
        for (pa, pb, no, f, other) in piece["labels"]:
            if tab_owner[no] != f:
                continue
            tab = _tab_polygon(pa, pb, piece["outline"])
            if tab is None or occupied[pi].intersection(tab.buffer(-0.05)).area > 0.05:
                # try the other side
                tab_owner[no] = other
                continue
            occupied[pi] = occupied[pi].union(tab)
            piece["tabs"].append((tab, no))
    # second pass for tabs that moved to the other side
    for pi, piece in enumerate(pieces):
        for (pa, pb, no, f, other) in piece["labels"]:
            if tab_owner[no] != f or any(n == no for _, n in piece["tabs"]):
                continue
            tab = _tab_polygon(pa, pb, piece["outline"], height=TAB_H * 0.6)
            if tab is not None:
                occupied[pi] = occupied[pi].union(tab)
                piece["tabs"].append((tab, no))
    for pi, piece in enumerate(pieces):
        piece["outline_full"] = occupied[pi]
    return pieces, next_no - 1


def _tab_polygon(pa, pb, outline, height=TAB_H):
    pa, pb = np.array(pa), np.array(pb)
    d = pb - pa
    L = np.linalg.norm(d)
    if L < 1.0:
        return None
    d /= L
    n = np.array([-d[1], d[0]])
    mid = (pa + pb) / 2
    from shapely.geometry import Point
    if outline.buffer(1e-6).contains(Point(*(mid + n * 0.3))):
        n = -n
    h = min(height, L * 0.45)
    inset = min(h, L * 0.3)
    return Polygon([tuple(pa), tuple(pb), tuple(pb + n * h - d * inset), tuple(pa + n * h + d * inset)])


# ------------------------------------------------------------- layout
def layout(pieces):
    """Rotate each piece to its tightest box and shelf-pack onto A4 pages."""
    placed = []
    for piece in pieces:
        geom = piece["outline_full"]
        rect = geom.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        ang = math.degrees(math.atan2(ys[1] - ys[0], xs[1] - xs[0]))
        # pick the rotation that fits the page, preferring the tightest box
        best = None
        for a in [ang, ang + 90] + list(range(0, 180, 15)):
            g = affinity.rotate(geom, -a, origin=(0, 0))
            minx, miny, maxx, maxy = g.bounds
            w, h = maxx - minx, maxy - miny
            score = (max(0, w - USABLE_W) + max(0, h - USABLE_H), w * h)
            if best is None or score < best[0]:
                best = (score, a, g.bounds)
        _, ang, (minx, miny, maxx, maxy) = best
        placed.append({"piece": piece, "angle": -ang, "w": maxx - minx,
                       "h": maxy - miny, "off": (-minx, -miny)})

    order = sorted(range(len(placed)), key=lambda i: -placed[i]["h"])
    pages = [[]]
    x = y = 0.0
    shelf_h = 0.0
    for i in order:
        p = placed[i]
        if x + p["w"] > USABLE_W:
            x, y = 0.0, y + shelf_h + GAP
            shelf_h = 0.0
        if y + p["h"] > USABLE_H and pages[-1]:
            pages.append([])
            x = y = shelf_h = 0.0
        p["at"] = (x, y)
        pages[-1].append(p)
        x += p["w"] + GAP
        shelf_h = max(shelf_h, p["h"])
    return pages


# --------------------------------------------------------------- render
def render_pdf(pages, colors, out_path, title, n_pairs, size_mm, face_count, preview_png=None):
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import Color

    c = canvas.Canvas(out_path, pagesize=(PAGE_W * MM, PAGE_H * MM))
    c.setTitle(title)

    # ---- cover / instructions page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(MARGIN * MM, (PAGE_H - 20) * MM, title)
    c.setFont("Helvetica", 10)
    lines = [
        f"Finished size: about {size_mm:.0f} mm on its longest side.   Faces: {face_count}.   "
        f"Pieces: {sum(len(p) for p in pages)}.   Sheets to print: {len(pages)} x {paper_name()}.",
        "",
        "Print at 100% / 'Actual size' (not 'Fit to page') on 160-200 gsm card.",
        "",
        "How to build",
        "1. Cut along every solid black line, including around the grey glue tabs.",
        "2. Score, then fold every dashed line:",
        "     long dashes  = valley fold (fold towards you, line ends up inside)",
        "     dash-dot     = mountain fold (fold away from you, line ends up outside)",
        "3. Every cut edge has a number. Glue each numbered tab under the edge with the same",
        "   number on another piece (or the same piece). Start with the largest piece.",
        "4. Close the model last through the edges that are easiest to reach.",
    ]
    yy = PAGE_H - 32
    for ln in lines:
        c.drawString(MARGIN * MM, yy * MM, ln)
        yy -= 5.2
    # legend
    yy -= 4
    for label, dash in (("cut", None), ("valley fold", (6, 3)), ("mountain fold", (6, 2, 1, 2))):
        c.setLineWidth(0.8)
        if dash:
            c.setDash(list(dash))
        else:
            c.setDash()
        c.line(MARGIN * MM, yy * MM, (MARGIN + 20) * MM, yy * MM)
        c.setDash()
        c.drawString((MARGIN + 24) * MM, (yy - 1.2) * MM, label)
        yy -= 6
    if preview_png:
        try:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(preview_png)
            iw, ih = img.getSize()
            maxw, maxh = USABLE_W, yy - MARGIN - 6
            s = min(maxw / iw, maxh / ih)
            c.drawImage(img, MARGIN * MM, (yy - 4 - ih * s) * MM, iw * s * MM, ih * s * MM,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
    c.showPage()

    for pno, page in enumerate(pages, 1):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(MARGIN * MM, (PAGE_H - MARGIN - 4) * MM, f"{title}  -  sheet {pno}/{len(pages)}")
        ox, oy = MARGIN, MARGIN
        for p in page:
            ang = p["angle"]
            th = math.radians(ang)
            cs, sn = math.cos(th), math.sin(th)
            dx, dy = p["off"]
            ax, ay = p["at"]

            def T(pt):
                x, y = pt
                xr, yr = x * cs - y * sn + dx + ax + ox, x * sn + y * cs + dy + ay + oy
                return xr * MM, yr * MM

            piece = p["piece"]
            # tabs
            for tab, no in piece["tabs"]:
                pts = [T(q) for q in list(tab.exterior.coords)[:-1]]
                path = c.beginPath()
                path.moveTo(*pts[0])
                for q in pts[1:]:
                    path.lineTo(*q)
                path.close()
                c.setFillColor(Color(0.85, 0.85, 0.85))
                c.setStrokeColor(Color(0, 0, 0))
                c.setLineWidth(0.5)
                c.setDash()
                c.drawPath(path, fill=1, stroke=1)
            # faces
            for f, pts in piece["faces"]:
                col = colors[f] / 255.0
                path = c.beginPath()
                tp = [T(q) for q in pts]
                path.moveTo(*tp[0])
                for q in tp[1:]:
                    path.lineTo(*q)
                path.close()
                c.setFillColor(Color(*col))
                c.setStrokeColor(Color(*col))
                c.setLineWidth(0.3)
                c.drawPath(path, fill=1, stroke=1)
            # folds
            c.setLineWidth(0.35)
            for pa, pb, kind in piece["folds"]:
                c.setStrokeColor(Color(0.15, 0.15, 0.15))
                c.setDash([6, 3] if kind == "valley" else [6, 2, 1, 2])
                c.line(*T(pa), *T(pb))
            c.setDash()
            # cuts
            c.setLineWidth(0.6)
            c.setStrokeColor(Color(0, 0, 0))
            for pa, pb in piece["cuts"]:
                c.line(*T(pa), *T(pb))
            # edge numbers, just inside each cut edge
            for pa, pb, no, f, other in piece["labels"]:
                A, B = np.array(T(pa)), np.array(T(pb))
                L = np.linalg.norm(B - A)
                if L < 3 * MM:
                    continue
                mid = (A + B) / 2
                d = (B - A) / L
                n = np.array([-d[1], d[0]])
                # move label to the inside of the face
                cen = np.mean([T(q) for q in dict(piece["faces"])[f]], axis=0)
                if np.dot(cen - mid, n) < 0:
                    n = -n
                fs = max(4.0, min(7.0, L / MM * 0.5))
                pos = mid + n * (fs * 0.9)
                col = colors[f] / 255.0
                lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                c.setFillColor(Color(1, 1, 1) if lum < 0.45 else Color(0, 0, 0))
                c.setFont("Helvetica", fs)
                c.saveState()
                c.translate(*pos)
                ang_deg = math.degrees(math.atan2(d[1], d[0]))
                if ang_deg > 90 or ang_deg < -90:
                    ang_deg += 180
                c.rotate(ang_deg)
                c.drawCentredString(0, -fs * 0.35, str(no))
                c.restoreState()
            for tab, no in piece["tabs"]:
                cen = tab.centroid
                c.setFillColor(Color(0.3, 0.3, 0.3))
                c.setFont("Helvetica", 4.5)
                x, y = T((cen.x, cen.y))
                c.drawCentredString(x, y - 1.5, str(no))
        c.showPage()
    c.save()


def paper_name():
    for k, v in PAPERS.items():
        if v == (PAGE_W, PAGE_H):
            return k
    return f"{PAGE_W:.0f}x{PAGE_H:.0f} mm"


# ---------------------------------------------------------------- main API
def photo_colour(photo_path):
    """Median colour of the object in a photo (pixels unlike the border background)."""
    from PIL import Image
    im = np.asarray(Image.open(photo_path).convert("RGB").resize((256, 256)), dtype=float)
    border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
    bg = np.median(border, axis=0)
    fg = im.reshape(-1, 3)[np.linalg.norm(im.reshape(-1, 3) - bg, axis=1) > 40]
    return np.median(fg, axis=0) if len(fg) > 50 else np.array([200.0, 200.0, 200.0])


def mesh_to_papercraft(mesh_path, out_pdf, faces=150, size_mm=150.0, title="Papercraft model",
                       preview_png=None, log=print, paper="A4"):
    set_paper(paper)
    mesh = load_mesh(mesh_path)
    log(f"loaded mesh: {len(mesh.faces)} faces")
    simple, colors = simplify(mesh, faces)
    # scale so the longest side is size_mm
    ext = simple.extents.max()
    simple.apply_scale(size_mm / ext)
    log(f"simplified to {len(simple.faces)} faces, watertight={simple.is_watertight}")
    if preview_png and (colors.max() < 25 or np.ptp(colors) < 3):
        # the 3D model came back without a usable texture: colour from the photo,
        # shaded a little by face direction so the folds stay readable
        base = photo_colour(preview_png)
        shade = 0.8 + 0.2 * np.abs(simple.face_normals[:, 1:2])
        colors = np.clip(base[None, :] * shade, 0, 255)
        log(f"model texture was empty; using photo colour {base.round().astype(int).tolist()}")
    islands, edge_info = unfold(simple)
    log(f"unfolded into {len(islands)} pieces")
    pieces, n_pairs = annotate(islands, edge_info, simple)
    pages = layout(pieces)
    render_pdf(pages, colors, out_pdf, title, n_pairs, size_mm, len(simple.faces), preview_png)
    log(f"wrote {out_pdf}: {len(pages)} {paper} sheet(s) + instructions page")
    return {"faces": len(simple.faces), "pieces": len(islands), "sheets": len(pages),
            "watertight": bool(simple.is_watertight)}
