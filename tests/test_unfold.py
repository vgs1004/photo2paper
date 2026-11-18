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


def test_pdf_written(tmp_path):
    path = tmp_path / "sphere.glb"
    trimesh.creation.icosphere(subdivisions=3).export(path)
    out = tmp_path / "sphere.pdf"
    info = U.mesh_to_papercraft(str(path), str(out), faces=100, log=lambda *a: None)
    assert out.stat().st_size > 1000
    assert info["sheets"] >= 1
