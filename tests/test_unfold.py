import collections
import os
import sys

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unfold as U  # noqa: E402

SHAPES = {
    "sphere": lambda: trimesh.creation.icosphere(subdivisions=4),
    "torus": lambda: trimesh.creation.torus(1.0, 0.4),
    "capsule": lambda: trimesh.creation.capsule(2.0, 0.6),
    "box": lambda: trimesh.creation.box((1, 2, 3)),
}


@pytest.mark.parametrize("name", SHAPES)
def test_unfold_is_buildable(name, tmp_path):
    mesh = SHAPES[name]()
    simple, _ = U.simplify(mesh, 150)
    simple.apply_scale(150 / simple.extents.max())
    assert simple.is_watertight

    islands, edge_info = U.unfold(simple)
    pieces, _ = U.annotate(islands, edge_info, simple)

    lengths = collections.defaultdict(list)
    tabs = collections.Counter()
    flat_area = 0.0
    for p in pieces:
        for pa, pb, no, _, _ in p["labels"]:
            lengths[no].append(np.linalg.norm(np.subtract(pa, pb)))
        for _, no in p["tabs"]:
            tabs[no] += 1
        polys = [Polygon(pts) for _, pts in p["faces"]]
        flat_area += sum(q.area for q in polys)
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert polys[i].intersection(polys[j]).area < 1e-3, "piece overlaps itself"

    for no, ls in lengths.items():
        assert len(ls) == 2, f"edge {no} has {len(ls)} sides"
        assert abs(ls[0] - ls[1]) < 1e-3
        assert tabs[no] == 1, f"edge {no} has {tabs[no]} tabs"
    assert flat_area == pytest.approx(simple.area, rel=1e-6)


def test_texture_is_printed(tmp_path):
    """A textured model gets its picture printed on the pieces, not flat colours."""
    from PIL import Image
    sphere = trimesh.creation.icosphere(subdivisions=4)
    v = sphere.vertices
    uv = np.c_[0.5 + np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi), 0.5 + np.arcsin(np.clip(v[:, 2], -1, 1)) / np.pi]
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:, :32] = (220, 30, 30)
    tex[:, 32:] = (30, 30, 220)
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    path = tmp_path / "ball.glb"
    sphere.export(path)
    out = tmp_path / "ball.pdf"
    U.mesh_to_papercraft(str(path), str(out), faces=100, log=lambda *a: None)
    import pymupdf
    doc = pymupdf.open(str(out))
    assert sum(len(p.get_images()) for p in doc) >= 1
    # both texture colours appear on the sheets
    pix = doc[1].get_pixmap(dpi=30)
    px = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].reshape(-1, 3)
    assert ((px[:, 0] > 150) & (px[:, 2] < 90)).any()
    assert ((px[:, 2] > 150) & (px[:, 0] < 90)).any()


def test_pdf_written(tmp_path):
    path = tmp_path / "sphere.glb"
    trimesh.creation.icosphere(subdivisions=3).export(path)
    out = tmp_path / "sphere.pdf"
    info = U.mesh_to_papercraft(str(path), str(out), faces=100, log=lambda *a: None)
    assert out.stat().st_size > 1000
    assert info["sheets"] >= 1


def test_simple_shapes_style(tmp_path):
    """Upright textured figure -> a few round parts with the picture printed on."""
    from PIL import Image
    import simple_shapes as SS
    # snowman-ish: two spheres stacked along Y, red/white texture by height
    a = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.6)
    b.apply_translation((0, 0, 1.5))
    m = trimesh.util.concatenate([a, b])
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))  # Z-up -> Y-up
    v = m.vertices
    uv = np.c_[0.5 + np.arctan2(v[:, 0], v[:, 2]) / (2 * np.pi), (v[:, 1] - v[:, 1].min()) / np.ptp(v[:, 1])]
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:32] = (230, 230, 230)   # top half of the image = high v = top of model
    tex[32:] = (200, 30, 30)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
    path = tmp_path / "snowman.glb"
    m.export(path)
    out = tmp_path / "snowman.pdf"
    info = SS.simple_papercraft(str(path), str(out), size_mm=120, max_parts=4, log=lambda *a: None)
    assert 2 <= info["parts"] <= 4
    assert info["pieces"] >= info["parts"]
    import pymupdf
    doc = pymupdf.open(str(out))
    assert len(doc) == info["sheets"] + 1
    assert sum(len(p.get_images()) for p in doc) >= info["pieces"]
