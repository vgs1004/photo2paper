"""Any photo -> 3D model -> printable fold-and-glue papercraft PDF.

Usage:
    python photo2paper.py headphones.jpg
    python photo2paper.py pen.png --faces 120 --size 180
    python photo2paper.py model.glb            # skip the AI step, unfold an existing 3D model

Step 1 sends the photo to a free hosted image-to-3D model on Hugging Face
(TRELLIS, falling back to Hunyuan3D-2). Step 2 runs locally: simplify,
unfold, add glue tabs, lay out on A4.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unfold import mesh_to_papercraft  # noqa: E402

MESH_EXT = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


def _token():
    """Optional Hugging Face token (free account) for a bigger daily GPU quota."""
    return os.environ.get("HF_TOKEN") or None


def _trellis(image_path, log):
    from gradio_client import Client, handle_file
    log("contacting TRELLIS (trellis-community/TRELLIS)...")
    client = Client("trellis-community/TRELLIS", verbose=False, token=_token())
    try:
        client.predict(api_name="/start_session")
    except Exception:
        pass
    result = client.predict(
        image=handle_file(image_path),
        multiimages=[],
        seed=0,
        ss_guidance_strength=7.5,
        ss_sampling_steps=12,
        slat_guidance_strength=3.0,
        slat_sampling_steps=12,
        multiimage_algo="stochastic",
        mesh_simplify=0.95,
        texture_size=1024,
        api_name="/generate_and_extract_glb",
    )
    glb = result[1] or result[2]
    return glb


def _hunyuan(image_path, log):
    from gradio_client import Client, handle_file
    log("contacting Hunyuan3D-2 (tencent/Hunyuan3D-2)...")
    client = Client("tencent/Hunyuan3D-2", verbose=False, token=_token())
    result = client.predict(
        caption=None,
        image=handle_file(image_path),
        mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
        steps=30, guidance_scale=5.0, seed=1234, octree_resolution=256,
        check_box_rembg=True, num_chunks=8000, randomize_seed=True,
        api_name="/shape_generation",
    )
    # shape only (their texture endpoint is unreliable); colour comes from the photo
    return result[0]


def photo_to_mesh(image_path, out_dir, log=print):
    errors = []
    for name, fn in (("TRELLIS", _trellis), ("Hunyuan3D-2", _hunyuan)):
        try:
            t = time.time()
            path = fn(image_path, log)
            if isinstance(path, dict):              # gradio sometimes wraps files
                path = path.get("value") or path.get("path") or path.get("name")
            if path and os.path.exists(path):
                ext = os.path.splitext(path)[1] or ".glb"
                dest = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_3d" + ext)
                shutil.copy(path, dest)
                log(f"{name} returned a 3D model in {time.time() - t:.0f}s -> {dest}")
                return dest
            errors.append(f"{name}: no file returned")
        except Exception as e:  # quota, queue, space asleep...
            msg = str(e).splitlines()[0][:200]
            log(f"{name} failed: {msg}")
            errors.append(f"{name}: {msg}")
    raise RuntimeError("Both hosted 3D models failed:\n  " + "\n  ".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="photo (jpg/png/webp) or an existing 3D model (glb/obj/ply/stl)")
    ap.add_argument("--faces", type=int, default=150,
                    help="paper faces; fewer = easier to build, more = more detail (default 150)")
    ap.add_argument("--size", type=float, default=150, help="longest side of finished model in mm (default 150)")
    ap.add_argument("--out", default=None, help="output PDF path")
    ap.add_argument("--title", default=None)
    ap.add_argument("--photo", default=None, help="with a 3D model input: the source photo (cover + colour fallback)")
    args = ap.parse_args()

    src = os.path.abspath(args.input)
    base = os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.dirname(os.path.abspath(args.out)) if args.out else os.path.dirname(src)
    out_pdf = args.out or os.path.join(out_dir, base + "_papercraft.pdf")
    title = args.title or base.replace("_", " ").replace("-", " ").title() + " papercraft"

    is_mesh = os.path.splitext(src)[1].lower() in MESH_EXT
    preview = None if is_mesh else src
    if is_mesh and args.photo:
        preview = os.path.abspath(args.photo)
    mesh_path = src if is_mesh else photo_to_mesh(src, out_dir)

    info = mesh_to_papercraft(mesh_path, out_pdf, faces=args.faces, size_mm=args.size,
                              title=title, preview_png=preview)
    print(f"\nDone: {out_pdf}")
    print(f"  {info['faces']} faces in {info['pieces']} pieces on {info['sheets']} A4 sheet(s)")
    if not info["watertight"]:
        print("  note: the 3D model has holes, so some edges are plain cut edges with no partner")


if __name__ == "__main__":
    main()
