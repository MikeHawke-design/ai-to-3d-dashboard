import numpy as np
import trimesh
from PIL import Image
from typing import Optional
from scipy.ndimage import gaussian_filter

class HeightmapGenerator:
    @staticmethod
    def create_heightmap_mesh(
        image: Image.Image,
        max_resolution: int = 128,
        height_scale: float = 10.0,
        base_thickness: float = 2.0,
        invert: bool = False,
        smooth_sigma: float = 0.0,
        depth_map: Optional[Image.Image] = None,
        adaptive_height: bool = True,
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
        
        if adaptive_height:
            z_normalized = z / 255.0
            z_contrast = np.std(z_normalized)
            adaptive_scale = height_scale * (1.0 + z_contrast * 0.5)
            z = z_normalized * adaptive_scale
        else:
            z = z / 255.0 * height_scale
        
        if smooth_sigma > 0:
            z = gaussian_filter(z, sigma=smooth_sigma)
        
        ny, nx = z.shape
        verts = []
        faces = []
        
        for y in range(ny):
            for x in range(nx):
                verts.append([float(x), float(y), float(z[y, x])])
        
        for y in range(ny - 1):
            for x in range(nx - 1):
                i0 = y * nx + x
                i1 = y * nx + (x + 1)
                i2 = (y + 1) * nx + x
                i3 = (y + 1) * nx + (x + 1)
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])
        
        verts = np.array(verts, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        
        top_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        
        if base_thickness > 0:
            bottom_verts = verts.copy()
            bottom_verts[:, 2] = -base_thickness
            offset = len(verts)
            all_verts = np.vstack([verts, bottom_verts])
            all_faces = list(faces)
            
            for y in range(ny - 1):
                for x in range(nx - 1):
                    i0 = offset + y * nx + x
                    i1 = offset + y * nx + (x + 1)
                    i2 = offset + (y + 1) * nx + x
                    i3 = offset + (y + 1) * nx + (x + 1)
                    all_faces.append([i0, i2, i1])
                    all_faces.append([i1, i2, i3])
            
            for x in range(nx - 1):
                t0 = x
                t1 = x + 1
                b0 = offset + x
                b1 = offset + x + 1
                all_faces.append([t0, t1, b0])
                all_faces.append([t1, b1, b0])
                
                t0b = (ny - 1) * nx + x
                t1b = (ny - 1) * nx + x + 1
                b0b = offset + (ny - 1) * nx + x
                b1b = offset + (ny - 1) * nx + x + 1
                all_faces.append([t0b, b0b, t1b])
                all_faces.append([t1b, b0b, b1b])
            
            for y in range(ny - 1):
                t0l = y * nx
                t1l = (y + 1) * nx
                b0l = offset + y * nx
                b1l = offset + (y + 1) * nx
                all_faces.append([t0l, b0l, t1l])
                all_faces.append([t1l, b0l, b1l])
                
                t0r = y * nx + (nx - 1)
                t1r = (y + 1) * nx + (nx - 1)
                b0r = offset + y * nx + (nx - 1)
                b1r = offset + (y + 1) * nx + (nx - 1)
                all_faces.append([t0r, t1r, b0r])
                all_faces.append([t1r, b1r, b0r])
            
            all_faces = np.array(all_faces, dtype=np.int32)
            mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
        else:
            mesh = top_mesh
        
        mesh.merge_vertices()
        trimesh.repair.fix_winding(mesh)
        return mesh
