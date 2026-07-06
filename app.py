import streamlit as st
import requests
import json
import io
import os
import tempfile
import time
import traceback
import math
import re
from typing import Optional, List, Dict, Any, Tuple

import trimesh
import numpy as np
from stl import mesh as stl_mesh, Mode as StlMode
from PIL import Image
from scipy.ndimage import gaussian_filter

from depth_optimizer import DepthPipeline, DepthOptimizer
from mesh_optimizer import MeshValidator

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="2D-to-3D Converter",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    defaults = {
        "current_mesh_data": None,
        "current_file_bytes": None,
        "current_file_type": "stl",
        "error_log": None,
        "openrouter_key": "",
        "ai_mode": "templates",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------
# EXTRUSION HELPERS
# ---------------------------------------------------------
def extrude_shape(points_2d: List[Tuple[float, float]], height: float = 5.0) -> trimesh.Trimesh:
    if len(points_2d) < 3:
        return trimesh.primitives.Box(extents=[10, 10, height])

    from shapely.geometry import Polygon as ShapelyPolygon
    poly = ShapelyPolygon(points_2d)
    if poly.area < 0.001:
        return trimesh.primitives.Box(extents=[10, 10, height])

    return trimesh.creation.extrude_polygon(poly, height=height)


# ---------------------------------------------------------
# TEMPLATE: GOTHIC WINDOW FRAME
# ---------------------------------------------------------

def gothic_window_frame(
    width: float = 60,
    height: float = 90,
    thickness: float = 5,
    frame_width: float = 6,
    trefoil: bool = True,
    base_height: float = 2,
) -> trimesh.Trimesh:
    hw = width / 2
    hh = height / 2

    def arch_points(rw: float, rh: float, segs: int = 32) -> List[Tuple[float, float]]:
        pts = []
        for i in range(segs + 1):
            t = i / segs
            angle = t * math.pi - math.pi / 2
            pts.append((rw * math.cos(angle), rh * math.sin(angle) + rh))
        for i in range(segs + 1):
            t = i / segs
            angle = math.pi / 2 - t * math.pi
            pts.append((-rw * math.cos(angle), rh * math.sin(angle) + rh))
        return pts

    outer = arch_points(hw, hh)
    iw = hw - frame_width
    ih = hh - frame_width * 0.6
    inner = arch_points(iw, ih)

    outer_shape = extrude_shape(outer, thickness)
    inner_shape = extrude_shape(inner, thickness)

    from trimesh.boolean import boolean_manifold
    frame = boolean_manifold([outer_shape, inner_shape], operation="difference", check_volume=False)

    if trefoil:
        r = frame_width * 0.4
        cy = hh * 0.6
        spacing = r * 1.2
        for dx in [-spacing, 0, spacing]:
            try:
                cutter = trimesh.primitives.Cylinder(radius=r, height=thickness * 2, sections=16)
                cutter.apply_translation([dx, cy, 0])
                frame = boolean_manifold([frame, cutter], operation="difference", check_volume=False)
            except Exception:
                pass

    if base_height > 0:
        base = trimesh.primitives.Box(extents=[width * 1.2, base_height, thickness])
        base.apply_translation([0, -base_height / 2, 0])
        try:
            frame = boolean_manifold([frame, base], operation="union", check_volume=False)
        except Exception:
            pass

    frame.merge_vertices()
    trimesh.repair.fix_winding(frame)
    return frame


# ---------------------------------------------------------
# TEMPLATE: GEAR
# ---------------------------------------------------------

def gear(
    outer_radius: float = 30,
    inner_radius: float = 20,
    teeth: int = 12,
    thickness: float = 8,
    hole_radius: float = 5,
) -> trimesh.Trimesh:
    pts = []
    n = teeth
    a_step = 2 * math.pi / n
    tooth_width = 0.6 * a_step * outer_radius
    gap_width = 0.4 * a_step * outer_radius
    for i in range(n):
        a = i * a_step
        pts.append((outer_radius * math.cos(a), outer_radius * math.sin(a)))
        a2 = a + a_step * 0.15
        pts.append((outer_radius * math.cos(a2), outer_radius * math.sin(a2)))
        a3 = a + a_step * 0.35
        pts.append((inner_radius * math.cos(a3), inner_radius * math.sin(a3)))
        a4 = a + a_step * 0.65
        pts.append((inner_radius * math.cos(a4), inner_radius * math.sin(a4)))
        a5 = a + a_step * 0.85
        pts.append((outer_radius * math.cos(a5), outer_radius * math.sin(a5)))
    m = extrude_shape(pts, thickness)
    if hole_radius > 0:
        try:
            hole = trimesh.primitives.Cylinder(radius=hole_radius, height=thickness, sections=16)
            from trimesh.boolean import boolean_manifold
            m = boolean_manifold([m, hole], operation="difference", check_volume=False)
        except Exception:
            pass
    return m


# ---------------------------------------------------------
# TEMPLATE: PLAQUE
# ---------------------------------------------------------

def plaque(
    width: float = 60,
    height: float = 40,
    thickness: float = 4,
    rounded: bool = True,
) -> trimesh.Trimesh:
    if rounded:
        verts = []
        n = 32
        for i in range(n):
            a = 2 * math.pi * i / n
            vx = width / 2 * math.cos(a)
            vy = height / 2 * math.sin(a)
            verts.append((vx, vy))
        return extrude_shape(verts, thickness)
    return trimesh.primitives.Box(extents=[width, height, thickness])


# ---------------------------------------------------------
# TEMPLATE: VASE
# ---------------------------------------------------------

def vase(height: float = 50, radius: float = 20, neck_radius: float = 8, thickness: float = 3) -> trimesh.Trimesh:
    from trimesh import boolean
    outer = trimesh.primitives.Cylinder(radius=radius, height=height, sections=32)
    top = trimesh.primitives.Cylinder(radius=neck_radius, height=height * 0.3, sections=32)
    top.apply_translation([0, 0, height * 0.35])
    try:
        body = boolean.union([outer, top], engine="manifold")
    except Exception:
        body = outer
    inner = trimesh.primitives.Cylinder(radius=radius - thickness, height=height * 0.8, sections=32)
    inner.apply_translation([0, 0, thickness])
    try:
        body = boolean.difference([body, inner], engine="manifold")
    except Exception:
        pass
    return body


# ---------------------------------------------------------
# PROMPT PARSER
# ---------------------------------------------------------

def parse_prompt(prompt: str) -> Tuple[str, Dict]:
    """Returns (template_name, params) based on keyword matching."""
    p = prompt.lower()

    templates = {
        "gothic": ("gothic_window", {}),
        "window": ("gothic_window", {}),
        "gear": ("gear", {}),
        "cog": ("gear", {}),
        "plaque": ("plaque", {}),
        "shield": ("plaque", {"width": 50, "height": 60}),
        "vase": ("vase", {}),
        "bowl": ("vase", {"height": 30, "radius": 25, "neck_radius": 20}),
        "box": ("box", {}),
        "cube": ("box", {}),
        "block": ("box", {}),
        "ring": ("ring", {}),
        "torus": ("ring", {}),
        "donut": ("ring", {"major": 20, "minor": 5}),
        "star": ("star", {}),
    }

    for keyword, (template, params) in templates.items():
        if keyword in p:
            return template, params

    return "box", {}


# ---------------------------------------------------------
# TEMPLATE DISPATCH
# ---------------------------------------------------------

TEMPLATES = {
    "gothic_window": lambda **kw: gothic_window_frame(
        width=kw.get("width", 60),
        height=kw.get("height", 90),
        thickness=kw.get("thickness", 5),
        frame_width=kw.get("frame_width", 6),
        trefoil=kw.get("trefoil", True),
    ),
    "gear": lambda **kw: gear(
        outer_radius=kw.get("outer_radius", 30),
        inner_radius=kw.get("inner_radius", 20),
        teeth=kw.get("teeth", 12),
        thickness=kw.get("thickness", 8),
    ),
    "plaque": lambda **kw: plaque(
        width=kw.get("width", 60),
        height=kw.get("height", 40),
        thickness=kw.get("thickness", 4),
    ),
    "vase": lambda **kw: vase(
        height=kw.get("height", 50),
        radius=kw.get("radius", 20),
        neck_radius=kw.get("neck_radius", 8),
    ),
    "box": lambda **kw: trimesh.primitives.Box(
        extents=[
            kw.get("width", 30),
            kw.get("depth", 30),
            kw.get("height", 30),
        ]
    ),
    "ring": lambda **kw: trimesh.creation.torus(
        major_radius=kw.get("major", 15),
        minor_radius=kw.get("minor", 4),
    ),
    "star": lambda **kw: extrude_shape(
        star_points(kw.get("points", 5), kw.get("outer", 25), kw.get("inner", 10)),
        kw.get("thickness", 5),
    ),
}


def star_points(n: int, outer: float, inner: float) -> List[Tuple[float, float]]:
    pts = []
    for i in range(n * 2):
        a = math.pi * i / n - math.pi / 2
        r = outer if i % 2 == 0 else inner
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


# ---------------------------------------------------------
# OPENROUTER LLM GENERATION
# ---------------------------------------------------------

OPENROUTER_FREE_CACHE: Optional[List[str]] = None
OPENROUTER_FREE_CACHE_TIME: float = 0


def fetch_free_models() -> List[str]:
    """Fetch current free models from OpenRouter. Cached for 5 min."""
    global OPENROUTER_FREE_CACHE, OPENROUTER_FREE_CACHE_TIME
    now = time.time()
    if OPENROUTER_FREE_CACHE and (now - OPENROUTER_FREE_CACHE_TIME) < 300:
        return OPENROUTER_FREE_CACHE
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        free = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", 1))
            completion_cost = float(pricing.get("completion", 1))
            if prompt_cost == 0 and completion_cost == 0:
                free.append(m["id"])
        free.sort()
        OPENROUTER_FREE_CACHE = free
        OPENROUTER_FREE_CACHE_TIME = now
        return free
    except Exception:
        return []


def generate_with_openrouter(prompt: str, api_key: str, model: str) -> trimesh.Trimesh:
    system_prompt = """You are a 3D geometry generator. Respond with ONLY a JSON object containing:
{
  "type": "primitive",
  "shape": "box|cylinder|sphere",
  "params": {}
}
OR for custom geometry:
{
  "type": "extrude",
  "points": [[x1,y1],[x2,y2],...],
  "height": float
}
No explanations, no markdown."""

    import time as _time
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate a 3D model of: {prompt}"},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            if resp.status_code == 429 and attempt < 2:
                retry_after = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                _time.sleep(retry_after)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            spec = json.loads(content)

            if spec.get("type") == "primitive":
                s = spec.get("shape", "box")
                p = spec.get("params", {})
                if s == "box":
                    return trimesh.primitives.Box(extents=[p.get("w", 30), p.get("d", 30), p.get("h", 30)])
                elif s == "cylinder":
                    return trimesh.primitives.Cylinder(radius=p.get("r", 15), height=p.get("h", 20), sections=32)
                elif s == "sphere":
                    return trimesh.primitives.Sphere(radius=p.get("r", 15), subdivisions=3)
            elif spec.get("type") == "extrude":
                pts = spec.get("points", [])
                h = spec.get("height", 10)
                if len(pts) >= 3:
                    return extrude_shape(pts, h)

            return trimesh.primitives.Box(extents=[30, 30, 30])

        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection failed: {e}"
            _time.sleep(2 ** attempt)
        except requests.exceptions.Timeout as e:
            last_error = f"Request timed out: {e}"
            _time.sleep(2 ** attempt)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON from AI: {e}"
            _time.sleep(1)
        except KeyError as e:
            last_error = f"Unexpected response format: missing {e}"
            _time.sleep(1)
        except Exception as e:
            last_error = str(e)
            _time.sleep(2 ** attempt)

    raise RuntimeError(f"OpenRouter generation failed after 3 attempts. Last error: {last_error}")


# ---------------------------------------------------------
# DEPTH MAP ENGINE
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI depth model (~700MB)...")
def load_depth_pipeline():
    from transformers import pipeline
    return pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")


def preprocess_image(image: Image.Image, target_res: int) -> Image.Image:
    """Prepare image for depth estimation with SculptOK-style optimizations."""
    import cv2
    
    # Normalize luminance and contrast
    img_arr = np.array(image)

    # Convert to grayscale intelligently based on image type
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        # RGB -> Convert using Luminance formula (0.299R + 0.587G + 0.114B)
        gray = np.dot(img_arr[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif len(img_arr.shape) == 2:
        gray = img_arr
    else:
        gray = img_arr[:, :, 0]

    # Adaptive histogram equalization for better contrast
    gray = gray.astype(np.float32)

    # Clip outliers and enhance local contrast
    p2, p98 = np.percentile(gray, [2, 98])
    gray = np.clip(gray, p2, p98)

    # Enhance contrast based on image type
    if np.std(gray) < 20:  # Low contrast images
        gray = (gray - np.mean(gray)) * 2 + np.mean(gray)

    # Apply slight unsharp mask to preserve edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    unsharp = gray - blurred
    gray = np.clip(gray + unsharp * 0.3, 0, 255)

    # Final resize with smooth interpolation
    result = Image.fromarray(np.uint8(gray), mode="L")
    return result.resize((target_res, target_res), Image.LANCZOS)


def multiscale_depth_estimation(image: Image.Image, target_res: int, levels: int = 4) -> List[np.ndarray]:
    """Multi-scale depth: runs AI depth at 4 input resolutions, returns numpy arrays."""
    pipe = load_depth_pipeline()
    depths = []

    # 4 AI depth layers at scales: 100%, 85%, 70%, 55%
    scales = [1.0 - (i * 0.15) for i in range(levels)]
    for scale in scales:
        w = max(int(image.width * scale), 64)
        h = max(int(image.height * scale), 64)
        small = image.resize((w, h), Image.LANCZOS)
        result = pipe(small)
        depth = result["depth"].resize((target_res, target_res), Image.LANCZOS)
        depths.append(np.array(depth, dtype=np.float32))

    return depths


def generate_depth_map(
    image: Image.Image,
    resolution: int = 256,
    detail_boost: float = 0.3,
    smooth_edges: float = 0.0,
    clahe_clip: float = 2.0,
) -> Image.Image:
    """4-layer AI depth estimation with detail-aware blending + full post-processing pipeline."""
    import cv2

    # 1. Multi-scale depth: 4 AI depth layers at 100%, 85%, 70%, 55% resolution
    with st.spinner("Running 4-layer multi-scale depth analysis..."):
        depth_layers = multiscale_depth_estimation(image, resolution, levels=4)

    # 2. Detail-aware blending (Laplacian-based weights — higher detail = higher weight)
    combined = DepthOptimizer.multi_scale_blend(depth_layers)

    # 3. Full post-processing pipeline (outliers → bilateral → morph → edge-aware → CLAHE → sharpen)
    pipeline = DepthPipeline()
    combined = pipeline.process(
        combined,
        remove_outliers=True,
        bilateral_denoise=True,
        edge_aware_smooth=True,
        clahe_enhance=True,
        sharpen=True,
        morph_close=True,
    )

    # 4. User-requested detail boost (unsharp mask on top of pipeline output)
    if detail_boost > 0:
        combined = np.clip(combined, 0, 255).astype(np.float32)
        blurred = cv2.GaussianBlur(combined, (7, 7), 1.5)
        detail = combined - blurred
        combined = combined + detail * detail_boost

    # 5. Normalize to 0-255
    combined = np.clip(combined, 0, 255).astype(np.float32)
    mn, mx = combined.min(), combined.max()
    if mx > mn:
        combined = (combined - mn) / (mx - mn) * 255.0

    return Image.fromarray(np.uint8(combined), mode="L")


# ---------------------------------------------------------
# HEIGHTMAP ENGINE
# ---------------------------------------------------------


def image_to_heightmap(
    image: Image.Image,
    max_resolution: int = 128,
    height_scale: float = 10.0,
    base_thickness: float = 2.0,
    invert: bool = False,
    smooth_sigma: float = 0.0,
    depth_map: Optional[Image.Image] = None,
) -> trimesh.Trimesh:
    if depth_map is not None:
        img = depth_map.convert("L")
    else:
        img = image.convert("L")

    w, h = img.size
    aspect = w / h
    if w > h:
        nw = max_resolution
        nh = int(max_resolution / aspect)
    else:
        nh = max_resolution
        nw = int(max_resolution * aspect)
    nw = max(nw, 2)
    nh = max(nh, 2)
    img = img.resize((nw, nh), Image.LANCZOS)

    z = np.array(img, dtype=np.float64)
    if invert:
        z = 255.0 - z
    z = z / 255.0 * height_scale

    if smooth_sigma > 0:
        z = gaussian_filter(z, sigma=smooth_sigma)

    ny, nx = z.shape

    # Vectorized vertex generation via meshgrid
    xGrid, yGrid = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    verts_top = np.stack([xGrid.ravel(), yGrid.ravel(), z.ravel()], axis=1)

    # Vectorized face generation
    yy, xx = np.mgrid[0 : ny - 1, 0 : nx - 1]
    i0 = (yy * nx + xx).ravel()
    i1 = (yy * nx + xx + 1).ravel()
    i2 = ((yy + 1) * nx + xx).ravel()
    i3 = ((yy + 1) * nx + xx + 1).ravel()
    faces_top = np.column_stack([
        np.stack([i0, i1, i2], axis=1),
        np.stack([i1, i3, i2], axis=1),
    ]).reshape(-1, 3)

    if base_thickness > 0:
        verts_bottom = verts_top.copy()
        verts_bottom[:, 2] = -base_thickness
        offset = len(verts_top)
        all_verts = np.vstack([verts_top, verts_bottom])

        # Bottom face (reversed winding)
        faces_bottom = faces_top[:, ::-1] + offset

        # Side faces — top edge
        t_top = np.arange(nx - 1)
        b_top = offset + t_top
        t_next = np.arange(1, nx)
        b_next = offset + t_next
        sides_top = np.column_stack([
            t_top, t_next, b_top,
            t_next, b_next, b_top,
        ]).reshape(-1, 3)

        # Side faces — bottom edge
        t0b = (ny - 1) * nx + np.arange(nx - 1)
        t1b = (ny - 1) * nx + np.arange(1, nx)
        b0b = offset + t0b
        b1b = offset + t1b
        sides_bottom = np.column_stack([
            t0b, b0b, t1b,
            t1b, b0b, b1b,
        ]).reshape(-1, 3)

        # Side faces — left edge
        t0l = np.arange(ny - 1) * nx
        t1l = np.arange(1, ny) * nx
        b0l = offset + t0l
        b1l = offset + t1l
        sides_left = np.column_stack([
            t0l, b0l, t1l,
            t1l, b0l, b1l,
        ]).reshape(-1, 3)

        # Side faces — right edge
        t0r = np.arange(ny - 1) * nx + (nx - 1)
        t1r = np.arange(1, ny) * nx + (nx - 1)
        b0r = offset + t0r
        b1r = offset + t1r
        sides_right = np.column_stack([
            t0r, t1r, b0r,
            t1r, b1r, b0r,
        ]).reshape(-1, 3)

        all_faces = np.vstack([faces_top, faces_bottom, sides_top, sides_bottom, sides_left, sides_right]).astype(np.int32)
        mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    else:
        mesh = trimesh.Trimesh(vertices=verts_top, faces=faces_top.astype(np.int32), process=False)

    mesh.merge_vertices()
    trimesh.repair.fix_winding(mesh)
    return mesh


# ---------------------------------------------------------
# EXPORT
# ---------------------------------------------------------

def export_mesh_bytes(t_mesh: trimesh.Trimesh, file_type: str) -> Tuple[bytes, dict]:
    """Validate + repair mesh, then export. Returns (bytes, quality_report)."""
    t_mesh, quality = MeshValidator.validate_and_repair(t_mesh, auto_repair=True)

    if file_type == "stl":
        numpy_stl_mesh = stl_mesh.Mesh(
            np.zeros(t_mesh.faces.shape[0], dtype=stl_mesh.Mesh.dtype)
        )
        for i, f in enumerate(t_mesh.faces):
            for j in range(3):
                numpy_stl_mesh.vectors[i][j] = t_mesh.vertices[f[j], :]
        buffer = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=True) as tmp:
            numpy_stl_mesh.save(tmp.name, fh=buffer, mode=StlMode.BINARY)
        return buffer.getvalue(), quality
    elif file_type == "obj":
        obj_str = trimesh.exchange.obj.export_obj(t_mesh)
        return obj_str.encode("utf-8"), quality
    raise ValueError("Unsupported file type")


# ---------------------------------------------------------
# 3D VIEWER
# ---------------------------------------------------------

def threejs_viewer_html(vertices: List[List[float]], faces: List[List[int]], auto_fit: bool = True) -> str:
    v_str = json.dumps(vertices)
    f_str = json.dumps(faces)

    # Auto-fit: compute bounding box center and extent for camera placement
    if auto_fit and vertices:
        verts_arr = np.array(vertices)
        center = verts_arr.mean(axis=0).tolist()
        extent = (verts_arr.max(axis=0) - verts_arr.min(axis=0)).max()
        cam_dist = max(extent * 1.5, 20)
    else:
        center = [0, 0, 0]
        cam_dist = 55

    return f"""
<!DOCTYPE html>
<html><head><style>
body {{ margin:0; overflow:hidden; background:#1a1a2e; }}
canvas {{ display:block; }}
#info {{ position:absolute; top:10px; left:10px; color:#8888aa; font:13px monospace; pointer-events:none; }}
</style></head><body>
<div id="info"></div>
<script type="importmap">{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set({center[0]} + {cam_dist}, {center[1]} + {cam_dist * 0.7}, {center[2]} + {cam_dist});
camera.lookAt({center[0]}, {center[1]}, {center[2]});
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.15;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;
controls.target.set({center[0]}, {center[1]}, {center[2]});
const ambient = new THREE.AmbientLight(0x404060, 1.5);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 2);
dir.position.set({center[0] + 10}, {center[1] + 20}, {center[2] + 10});
scene.add(dir);
scene.add(new THREE.DirectionalLight(0x8888ff, 0.5));
const vertices = {v_str};
const faces = {f_str};
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(vertices.flat());
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
const idx = new Uint32Array(faces.flat());
geo.setIndex(new THREE.BufferAttribute(idx, 1));
geo.computeVertexNormals();
const mat = new THREE.MeshPhysicalMaterial({{ color: 0x4fc3f7, metalness: 0.1, roughness: 0.4, side: THREE.DoubleSide }});
const mesh = new THREE.Mesh(geo, mat);
scene.add(mesh);
// Wireframe only for small meshes
if (vertices.length < 50000) {{
  const wire = new THREE.WireframeGeometry(geo);
  const line = new THREE.LineSegments(wire, new THREE.LineBasicMaterial({{ color: 0x1a6b8a, transparent: true, opacity: 0.12 }}));
  mesh.add(line);
}}
const grid = new THREE.GridHelper(80, 20, 0x444466, 0x333355);
grid.position.y = {center[1] - 5};
scene.add(grid);
function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
animate();
window.addEventListener('resize', () => {{ camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }});
document.addEventListener('click', () => {{ controls.autoRotate = false; }});
</script></body></html>
"""


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.header("Settings")
        st.subheader("Export")
        st.session_state.current_file_type = st.radio(
            "Format", ["stl", "obj"], horizontal=True, index=0
        )

        st.divider()
        st.subheader("OpenRouter AI")

        or_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.openrouter_key,
            placeholder="sk-or-v1-...",
            help="Get your key at https://openrouter.ai/keys",
        )
        if or_key != st.session_state.openrouter_key:
            st.session_state.openrouter_key = or_key

        if or_key:
            free_models = fetch_free_models()
            if free_models:
                st.selectbox("Free Model", free_models, index=0, key="or_model")
            else:
                st.info("Could not fetch model list.")
                st.text_input("Model ID", value="google/gemini-2.0-flash-001", key="or_model_manual")


# ---------------------------------------------------------
# HEIGHTMAP TAB
# ---------------------------------------------------------

def render_heightmap_tab():
    st.subheader("Upload a 2D image to convert into a 3D printable model")

    uploaded = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp"],
    )
    if uploaded:
        st.image(uploaded, caption="Source", width=200)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        resolution = st.slider("Resolution", 32, 256, 128)
    with col2:
        height_val = st.slider("Height (mm)", 1.0, 50.0, 10.0)
    with col3:
        base = st.slider("Base (mm)", 0.0, 10.0, 2.0)
    with col4:
        smooth = st.slider("Smooth", 0.0, 5.0, 0.5)

    col_depth, col_inv, col_gen = st.columns([2, 1, 3])
    with col_depth:
        use_depth = st.checkbox("AI depth map", value=True, help="Generate depth map with AI for realistic 3D mapping.")
    with col_inv:
        invert = st.checkbox("Invert height", value=False)
    with col_gen:
        generate = st.button(
            "Generate 3D Model", type="primary", use_container_width=True,
            disabled=not uploaded, key="gen_heightmap",
        )

    if use_depth:
        with st.expander("Depth settings", expanded=True):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.slider("Detail boost", 0.0, 3.0, 0.5, 0.1, key="dm_detail", help="Enhance fine details (unsharp mask)")
                st.slider("Local contrast", 0.0, 5.0, 2.0, 0.1, key="dm_clahe", help="CLAHE - reveals hidden detail in dark/bright areas")
            with col_d2:
                st.slider("Edge smooth", 0.0, 3.0, 0.3, 0.1, key="dm_smooth", help="Smooth noise while preserving edges (bilateral)")
                st.slider("Depth intensity", 0.0, 3.0, 1.0, 0.1, key="dm_intensity", help="Overall depth strength multiplier")

    if generate and uploaded:
        st.session_state.error_log = None
        st.session_state.current_mesh_data = None
        st.session_state.current_file_bytes = None

        with st.spinner("Building 3D model..."):
            try:
                img = Image.open(uploaded)
                depth_map = None
                if use_depth:
                    with st.spinner("Generating AI depth map..."):
                        detail = st.session_state.get("dm_detail", 0.5)
                        esmooth = st.session_state.get("dm_smooth", 0.3)
                        intensity = st.session_state.get("dm_intensity", 1.0)
                        clahe = st.session_state.get("dm_clahe", 2.0)
                        depth_map = generate_depth_map(
                            img, resolution=resolution * 2,
                            detail_boost=detail, smooth_edges=esmooth,
                            clahe_clip=clahe,
                        )
                        st.image(depth_map, caption="AI depth map", width=250)
                t_mesh = image_to_heightmap(
                    img, max_resolution=resolution, height_scale=height_val * intensity,
                    base_thickness=base, invert=invert, smooth_sigma=smooth,
                    depth_map=depth_map,
                )
                file_bytes, mesh_quality = export_mesh_bytes(t_mesh, st.session_state.current_file_type)
                st.session_state.current_mesh_data = t_mesh
                st.session_state.current_file_bytes = file_bytes
                st.session_state.mesh_quality = mesh_quality
                st.rerun()
            except Exception as e:
                st.session_state.error_log = f"Heightmap error: {e}"
                st.rerun()


# ---------------------------------------------------------
# AI (TEMPLATE) TAB
# ---------------------------------------------------------

TEMPLATE_NAMES = {
    "gothic_window": "Gothic Window Frame",
    "gear": "Gear / Cog",
    "plaque": "Plaque / Shield",
    "vase": "Vase / Bowl",
    "box": "Box / Cube",
    "ring": "Ring / Torus / Donut",
    "star": "Star",
}

EXAMPLE_PROMPTS = [
    "a gothic window frame with pointed arch and trefoil details",
    "a gear with 12 teeth, 30mm radius, 8mm thick",
    "a rectangular plaque 60x40x4mm with rounded corners",
    "a vase 50mm tall with narrow neck",
    "a 5-pointed star 50mm across, 5mm thick",
    "a ring 30mm outer diameter, 5mm thick",
]


def render_ai_tab():
    mode = st.radio(
        "Generation mode:",
        ["Free Templates", "OpenRouter (AI)"],
        horizontal=True,
        index=0,
        key="ai_mode_selector",
    )

    if mode == "Free Templates":
        render_template_tab()
    elif mode == "OpenRouter (AI)":
        render_openrouter_tab()


def render_template_tab():
    st.subheader("Generate a 3D model from a description")
    st.caption("100% free — no API key needed. Uses built-in shape templates.")

    col_e, _ = st.columns([1, 2])
    with col_e:
        example_choice = st.selectbox("Try an example:", [""] + EXAMPLE_PROMPTS, key="tmpl_example")

    prompt = st.text_area(
        "Describe your shape:",
        height=100,
        value=example_choice if example_choice else "",
        placeholder="a gothic window frame with pointed arch...",
        key="tmpl_prompt",
    )

    if prompt:
        template_name, _ = parse_prompt(prompt)
        display = TEMPLATE_NAMES.get(template_name, "Box")
        st.caption(f"Detected: {display}")

    if st.button("Generate 3D Model", type="primary", use_container_width=True, disabled=not prompt, key="gen_tmpl"):
        st.session_state.error_log = None
        st.session_state.current_mesh_data = None
        st.session_state.current_file_bytes = None

        with st.spinner("Building 3D model..."):
            try:
                template_name, params = parse_prompt(prompt)
                builder = TEMPLATES.get(template_name)
                if builder:
                    t_mesh = builder(**params)
                else:
                    t_mesh = trimesh.primitives.Box(extents=[30, 30, 30])

                file_bytes, mesh_quality = export_mesh_bytes(t_mesh, st.session_state.current_file_type)
                st.session_state.current_mesh_data = t_mesh
                st.session_state.current_file_bytes = file_bytes
                st.session_state.mesh_quality = mesh_quality
                st.rerun()
            except Exception as e:
                st.session_state.error_log = f"Generation error: {e}\n{traceback.format_exc()}"
                st.rerun()


def render_openrouter_tab():
    st.subheader("Generate with OpenRouter AI")
    st.caption("Select a free model and enter your API key in the sidebar.")

    if not st.session_state.openrouter_key:
        st.warning("Enter your OpenRouter API key in the sidebar first.")
        return

    model = st.session_state.get("or_model", "") or st.session_state.get("or_model_manual", "google/gemini-2.0-flash-001")
    if model:
        st.info(f"Model: {model}")

    prompt = st.text_area(
        "Describe your 3D shape:",
        height=100,
        placeholder="a decorative vase with fluted edges...",
        key="or_prompt",
    )

    if st.button("Generate with OpenRouter", type="primary", use_container_width=True, disabled=not prompt, key="gen_or"):
        st.session_state.error_log = None
        st.session_state.current_mesh_data = None
        st.session_state.current_file_bytes = None

        with st.spinner("Asking AI to generate geometry..."):
            try:
                t_mesh = generate_with_openrouter(prompt, st.session_state.openrouter_key, model)
                file_bytes, mesh_quality = export_mesh_bytes(t_mesh, st.session_state.current_file_type)
                st.session_state.current_mesh_data = t_mesh
                st.session_state.current_file_bytes = file_bytes
                st.session_state.mesh_quality = mesh_quality
                st.rerun()
            except Exception as e:
                st.session_state.error_log = f"OpenRouter error: {e}\n{traceback.format_exc()}"
                st.rerun()


# ---------------------------------------------------------
# 3D PREVIEW & DOWNLOAD
# ---------------------------------------------------------

def render_preview():
    mesh = st.session_state.current_mesh_data
    if mesh is None:
        return

    st.divider()
    st.subheader("3D Preview")

    verts = mesh.vertices.tolist()
    faces = mesh.faces.tolist()
    num_verts = len(verts)

    # Vertex warning + auto-simplify for large meshes
    display_mesh = mesh
    if num_verts > 100000:
        st.warning(f"Large mesh ({num_verts:,} vertices). Auto-simplifying for preview...")
        try:
            display_mesh = mesh.simplify_quadric_decimation(50000)
            verts = display_mesh.vertices.tolist()
            faces = display_mesh.faces.tolist()
            st.caption(f"Simplified to {len(verts):,} vertices for preview (full mesh preserved in export)")
        except Exception:
            st.caption("Simplification failed — showing full mesh (may be slow)")

    html = threejs_viewer_html(verts, faces)
    st.components.v1.html(html, height=500)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vertices", num_verts)
    m2.metric("Faces", len(mesh.faces))
    m3.metric("Volume (mm)", f"{mesh.volume:.1f}" if mesh.is_volume else "N/A")
    m4.metric("Watertight", "Yes" if mesh.is_watertight else "No")

    # Mesh quality report from MeshValidator
    quality = st.session_state.get("mesh_quality")
    if quality:
        with st.expander("Mesh Quality Report"):
            before = quality.get("before", {})
            after = quality.get("after", {})
            col_b, col_a = st.columns(2)
            with col_b:
                st.caption("Before repair")
                st.json(before)
            with col_a:
                st.caption("After repair")
                st.json(after)

    with st.expander("Bounds & Dimensions"):
        bounds = mesh.bounds
        dims = bounds[1] - bounds[0]
        st.json({
            "Min": bounds[0].tolist(),
            "Max": bounds[1].tolist(),
            "Size (X,Y,Z mm)": [round(d, 2) for d in dims.tolist()],
        })

    file_ext = st.session_state.current_file_type
    mime = "application/sla" if file_ext == "stl" else "model/obj"
    filename = f"model.{file_ext}"

    st.download_button(
        label=f"Download {filename.upper()}",
        data=st.session_state.current_file_bytes,
        file_name=filename,
        mime=mime,
        use_container_width=True,
        type="primary",
    )

    st.info("Drag the .stl into Bambu Studio / PrusaSlicer / Cura.")


# ---------------------------------------------------------
# ERROR DISPLAY
# ---------------------------------------------------------

def render_error():
    if st.session_state.error_log:
        st.error(st.session_state.error_log)
        if st.button("Clear error", key="clear_err"):
            st.session_state.error_log = None
            st.rerun()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    st.title("2D to 3D Converter")
    st.caption("Upload a 2D image for instant 3D heightmap, or describe a shape to generate it.")

    tab1, tab2 = st.tabs(["Image to 3D", "Describe & Generate"])

    with tab1:
        render_heightmap_tab()

    with tab2:
        render_ai_tab()

    render_error()
    render_preview()


if __name__ == "__main__":
    render_sidebar()
    main()
