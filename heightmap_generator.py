"""
Advanced heightmap generation with displacement mapping and adaptive scaling.
"""

import numpy as np
import trimesh
from PIL import Image
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter


class HeightmapGenerator:
    """Advanced heightmap to 3D mesh conversion."""
    
    @staticmethod
    def analyze_image_content(image_array: np.ndarray) -> dict:
        """Analyze image to determine optimal conversion parameters."""
        # Calculate statistics
        mean_val = np.mean(image_array)
        std_val = np.std(image_array)
        min_val = np.min(image_array)
        max_val = np.max(image_array)
        
        # Detect contrast level
        contrast = (max_val - min_val) / 255.0 if max_val > min_val else 0.0
        
        # Detect if image is mostly dark or bright
        brightness = mean_val / 255.0
        
        # Compute edge density
        from scipy.ndimage import laplace
        edges = np.abs(laplace(image_array))
        edge_density = np.mean(edges) / 255.0
        
        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "contrast": contrast,
            "brightness": brightness,
            "edge_density": edge_density,
            "dynamic_range": max_val - min_val,
        }
    
    @staticmethod
    def adaptive_height_scaling(image_array: np.ndarray, base_height: float = 10.0) -> float:
        """Automatically adjust height scaling based on image content."""
        analysis = HeightmapGenerator.analyze_image_content(image_array)
        
        # Higher contrast = less height needed for detail preservation
        contrast_factor = 1.0 - (analysis["contrast"] * 0.3)
        
        # Higher edge density = more height needed for fine details
        edge_factor = 1.0 + (analysis["edge_density"] * 0.5)
        
        # Adjust for brightness (avoid extreme values)
        brightness_factor = 1.0 if 0.3 < analysis["brightness"] < 0.7 else 0.8
        
        optimal_height = base_height * contrast_factor * edge_factor * brightness_factor
        return np.clip(optimal_height, base_height * 0.5, base_height * 2.0)
    
    @staticmethod
    def apply_displacement_map(
        vertices: np.ndarray,
        displacement: np.ndarray,
        strength: float = 1.0,
        grid_shape: Tuple[int, int] = None
    ) -> np.ndarray:
        """Apply displacement map to vertices for finer detail."""
        if grid_shape is None:
            grid_shape = displacement.shape
        
        ny, nx = grid_shape
        displaced = vertices.copy()
        
        # Map each vertex to displacement grid coordinates
        for i, (x, y, z) in enumerate(vertices):
            # Normalize coordinates to grid
            grid_x = int(np.clip(x, 0, nx - 1))
            grid_y = int(np.clip(y, 0, ny - 1))
            
            # Get displacement value (normalized to 0-1)
            disp_value = displacement[grid_y, grid_x] / 255.0
            
            # Apply displacement in Z direction
            displaced[i, 2] += disp_value * strength
        
        return displaced
    
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
        displacement_strength: float = 0.5,
    ) -> trimesh.Trimesh:
        """Create high-quality heightmap mesh with multiple layers of detail."""
        
        # Use depth map if provided, otherwise use image luminance
        if depth_map is not None:
            img = depth_map.convert("L")
        else:
            img = image.convert("L")
        
        # Resize maintaining aspect ratio
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
        
        # Convert to array
        z = np.array(img, dtype=np.float64)
        
        # Invert if needed
        if invert:
            z = 255.0 - z
        
        # Analyze content for adaptive height
        if adaptive_height:
            optimal_height = HeightmapGenerator.adaptive_height_scaling(z, height_scale)
        else:
            optimal_height = height_scale
        
        # Normalize to height scale
        z = z / 255.0 * optimal_height
        
        # Apply smoothing if requested
        if smooth_sigma > 0:
            z = gaussian_filter(z, sigma=smooth_sigma)
        
        ny, nx = z.shape
        
        # Create vertices for top surface
        verts = []
        for y in range(ny):
            for x in range(nx):
                verts.append([float(x), float(y), float(z[y, x])])
        
        verts = np.array(verts, dtype=np.float64)
        
        # Create faces for top surface
        faces = []
        for y in range(ny - 1):
            for x in range(nx - 1):
                i0 = y * nx + x
                i1 = y * nx + (x + 1)
                i2 = (y + 1) * nx + x
                i3 = (y + 1) * nx + (x + 1)
                faces.append([i0, i1, i2])
                faces.append([i1, i3, i2])
        
        # Create base
        if base_thickness > 0:
            offset = len(verts)
            bottom_verts = verts.copy()
            bottom_verts[:, 2] = -base_thickness
            verts = np.vstack([verts, bottom_verts])
            
            # Bottom face (reversed winding)
            for y in range(ny - 1):
                for x in range(nx - 1):
                    i0 = offset + y * nx + x
                    i1 = offset + y * nx + (x + 1)
                    i2 = offset + (y + 1) * nx + x
                    i3 = offset + (y + 1) * nx + (x + 1)
                    faces.append([i0, i2, i1])
                    faces.append([i1, i2, i3])
            
            # Side faces
            # Front edge
            for x in range(nx - 1):
                t0 = x
                t1 = x + 1
                b0 = offset + x
                b1 = offset + x + 1
                faces.append([t0, t1, b0])
                faces.append([t1, b1, b0])
            
            # Back edge
            for x in range(nx - 1):
                t0b = (ny - 1) * nx + x
                t1b = (ny - 1) * nx + x + 1
                b0b = offset + (ny - 1) * nx + x
                b1b = offset + (ny - 1) * nx + x + 1
                faces.append([t0b, b0b, t1b])
                faces.append([t1b, b0b, b1b])
            
            # Left edge
            for y in range(ny - 1):
                t0l = y * nx
                t1l = (y + 1) * nx
                b0l = offset + y * nx
                b1l = offset + (y + 1) * nx
                faces.append([t0l, b0l, t1l])
                faces.append([t1l, b0l, b1l])
            
            # Right edge
            for y in range(ny - 1):
                t0r = y * nx + (nx - 1)
                t1r = (y + 1) * nx + (nx - 1)
                b0r = offset + y * nx + (nx - 1)
                b1r = offset + (y + 1) * nx + (nx - 1)
                faces.append([t0r, t1r, b0r])
                faces.append([t1r, b1r, b0r])
        
        faces = np.array(faces, dtype=np.int32)
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.merge_vertices()
        
        try:
            trimesh.repair.fix_winding(mesh)
        except Exception:
            pass
        
        return mesh
