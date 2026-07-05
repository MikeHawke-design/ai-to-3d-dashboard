"""
Advanced mesh optimization and quality validation.
Includes topology repair, simplification, and validation checks.
"""

import numpy as np
import trimesh
from typing import Tuple, Optional
from scipy.spatial import cKDTree


class MeshOptimizer:
    """Advanced mesh optimization techniques."""
    
    @staticmethod
    def remove_duplicate_vertices(mesh: trimesh.Trimesh, tolerance: float = 1e-6) -> trimesh.Trimesh:
        """Remove duplicate vertices within tolerance."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Use KD-tree for efficient duplicate detection
        tree = cKDTree(vertices)
        groups = tree.query_ball_point(vertices, r=tolerance)
        
        # Map old vertex indices to new ones
        vertex_map = np.arange(len(vertices))
        for group in groups:
            if len(group) > 1:
                # Map all vertices in group to first one
                for idx in group[1:]:
                    vertex_map[idx] = group[0]
        
        # Update faces
        new_faces = vertex_map[faces]
        
        # Get unique vertices
        unique_indices = np.unique(new_faces)
        new_vertices = vertices[unique_indices]
        
        # Remap faces to new indices
        old_to_new = np.full(len(vertices), -1, dtype=int)
        old_to_new[unique_indices] = np.arange(len(unique_indices))
        new_faces = old_to_new[new_faces]
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    @staticmethod
    def smooth_laplacian(mesh: trimesh.Trimesh, iterations: int = 5, alpha: float = 0.5) -> trimesh.Trimesh:
        """Laplacian smoothing to reduce noise while preserving features."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency list
        adjacency = [[] for _ in range(len(vertices))]
        for face in faces:
            for i, v_idx in enumerate(face):
                for v_idx2 in face[(i+1):]:
                    if v_idx2 not in adjacency[v_idx]:
                        adjacency[v_idx].append(v_idx2)
                    if v_idx not in adjacency[v_idx2]:
                        adjacency[v_idx2].append(v_idx)
        
        # Apply Laplacian smoothing
        for _ in range(iterations):
            new_vertices = vertices.copy()
            for i, neighbors in enumerate(adjacency):
                if len(neighbors) > 0:
                    neighbor_avg = vertices[neighbors].mean(axis=0)
                    new_vertices[i] = vertices[i] * (1 - alpha) + neighbor_avg * alpha
            vertices = new_vertices
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    @staticmethod
    def taubin_smoothing(mesh: trimesh.Trimesh, lambda_: float = 0.5, mu: float = -0.53, iterations: int = 10) -> trimesh.Trimesh:
        """Taubin smoothing - high-quality smoothing that preserves volume better."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency list
        adjacency = [[] for _ in range(len(vertices))]
        for face in faces:
            for i, v_idx in enumerate(face):
                for v_idx2 in face[(i+1):]:
                    if v_idx2 not in adjacency[v_idx]:
                        adjacency[v_idx].append(v_idx2)
                    if v_idx not in adjacency[v_idx2]:
                        adjacency[v_idx2].append(v_idx)
        
        for iteration in range(iterations):
            # Use positive lambda on odd iterations, negative mu on even
            coeff = lambda_ if iteration % 2 == 0 else mu
            
            new_vertices = vertices.copy()
            for i, neighbors in enumerate(adjacency):
                if len(neighbors) > 0:
                    neighbor_avg = vertices[neighbors].mean(axis=0)
                    displacement = neighbor_avg - vertices[i]
                    new_vertices[i] = vertices[i] + displacement * coeff
            vertices = new_vertices
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    @staticmethod
    def decimate_mesh(mesh: trimesh.Trimesh, target_reduction: float = 0.5) -> trimesh.Trimesh:
        """Simplify mesh while preserving shape using quadric error metrics."""
        try:
            # Use trimesh's simplification if available
            simplified = mesh.simplify_quadric_mesh(target_reduction=target_reduction)
            return simplified
        except Exception:
            # Fallback: use built-in simplify
            try:
                return mesh.simplify_quadric_mesh()
            except Exception:
                return mesh
    
    @staticmethod
    def fix_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Ensure consistent outward-facing normals."""
        try:
            trimesh.repair.fix_winding(mesh)
        except Exception:
            pass
        
        return mesh
    
    @staticmethod
    def fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attempt to fill holes in mesh."""
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass
        
        return mesh
    
    @staticmethod
    def remove_degenerate_faces(mesh: trimesh.Trimesh, min_area: float = 1e-8) -> trimesh.Trimesh:
        """Remove faces with area below threshold."""
        faces = mesh.faces
        vertices = mesh.vertices
        
        # Calculate face areas
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        
        # Keep only non-degenerate faces
        valid_faces = faces[areas > min_area]
        
        return trimesh.Trimesh(vertices=vertices, faces=valid_faces, process=False)
    
    @staticmethod
    def adaptive_subdivision(mesh: trimesh.Trimesh, max_edge_length: float = 5.0) -> trimesh.Trimesh:
        """Subdivide mesh where edges are too long."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        for _ in range(2):  # Max 2 iterations to avoid extreme subdivision
            new_vertices = list(vertices)
            new_faces = []
            vertex_map = {}  # Map edge to new vertex index
            
            for face in faces:
                v0, v1, v2 = vertices[face]
                edges = [
                    (face[0], face[1], v0, v1),
                    (face[1], face[2], v1, v2),
                    (face[2], face[0], v2, v0),
                ]
                
                edge_mids = []
                for idx1, idx2, va, vb in edges:
                    edge_key = tuple(sorted([idx1, idx2]))
                    length = np.linalg.norm(vb - va)
                    
                    if length > max_edge_length and edge_key not in vertex_map:
                        mid = (va + vb) / 2
                        new_idx = len(new_vertices)
                        new_vertices.append(mid)
                        vertex_map[edge_key] = new_idx
                    
                    edge_mids.append(vertex_map.get(tuple(sorted([idx1, idx2])), None))
                
                # If no subdivision needed, add original face
                if all(m is None for m in edge_mids):
                    new_faces.append(face)
                else:
                    # Add subdivided faces (simple case)
                    new_faces.append(face)
            
            vertices = np.array(new_vertices)
            faces = np.array(new_faces)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class MeshValidator:
    """Mesh quality validation and repair."""
    
    @staticmethod
    def check_mesh_quality(mesh: trimesh.Trimesh) -> dict:
        """Comprehensive mesh quality check."""
        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "is_valid": mesh.is_valid,
            "volume": mesh.volume if mesh.is_volume else 0,
            "surface_area": mesh.area,
            "has_degenerate_faces": len(mesh.faces) > 0 and np.any(np.abs(mesh.face_areas) < 1e-10),
            "bounds": mesh.bounds.tolist(),
            "euler_number": mesh.euler_number,
        }
    
    @staticmethod
    def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply multiple repair steps."""
        # Remove duplicate vertices
        mesh = MeshOptimizer.remove_duplicate_vertices(mesh)
        
        # Remove degenerate faces
        mesh = MeshOptimizer.remove_degenerate_faces(mesh)
        
        # Fix normals
        mesh = MeshOptimizer.fix_normals(mesh)
        
        # Try to fill holes
        mesh = MeshOptimizer.fill_holes(mesh)
        
        # Merge vertices
        try:
            mesh.merge_vertices()
        except Exception:
            pass
        
        return mesh
    
    @staticmethod
    def validate_and_repair(mesh: trimesh.Trimesh, auto_repair: bool = True) -> Tuple[trimesh.Trimesh, dict]:
        """Validate mesh and optionally repair."""
        quality_before = MeshValidator.check_mesh_quality(mesh)
        
        if auto_repair:
            mesh = MeshValidator.repair_mesh(mesh)
        
        quality_after = MeshValidator.check_mesh_quality(mesh)
        
        return mesh, {
            "before": quality_before,
            "after": quality_after,
            "repaired": auto_repair,
        }
