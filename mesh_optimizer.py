import numpy as np
import trimesh
from typing import Tuple, Optional
from scipy.spatial import cKDTree

class MeshOptimizer:
    @staticmethod
    def remove_duplicate_vertices(mesh: trimesh.Trimesh, tolerance: float = 1e-6) -> trimesh.Trimesh:
        vertices = mesh.vertices
        faces = mesh.faces
        tree = cKDTree(vertices)
        groups = tree.query_ball_point(vertices, r=tolerance)
        vertex_map = np.arange(len(vertices))
        for group in groups:
            if len(group) > 1:
                for idx in group[1:]:
                    vertex_map[idx] = group[0]
        new_faces = vertex_map[faces]
        unique_indices = np.unique(new_faces)
        new_vertices = vertices[unique_indices]
        old_to_new = np.full(len(vertices), -1, dtype=int)
        old_to_new[unique_indices] = np.arange(len(unique_indices))
        new_faces = old_to_new[new_faces]
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    @staticmethod
    def taubin_smoothing(mesh: trimesh.Trimesh, lambda_: float = 0.5, mu: float = -0.53, iterations: int = 10) -> trimesh.Trimesh:
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        adjacency = [[] for _ in range(len(vertices))]
        for face in faces:
            for i, v_idx in enumerate(face):
                for v_idx2 in face[(i+1):]:
                    if v_idx2 not in adjacency[v_idx]:
                        adjacency[v_idx].append(v_idx2)
                    if v_idx not in adjacency[v_idx2]:
                        adjacency[v_idx2].append(v_idx)
        for iteration in range(iterations):
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
    def remove_degenerate_faces(mesh: trimesh.Trimesh, min_area: float = 1e-8) -> trimesh.Trimesh:
        faces = mesh.faces
        vertices = mesh.vertices
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        valid_faces = faces[areas > min_area]
        return trimesh.Trimesh(vertices=vertices, faces=valid_faces, process=False)

class MeshValidator:
    @staticmethod
    def check_mesh_quality(mesh: trimesh.Trimesh) -> dict:
        return {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "is_valid": mesh.is_valid,
            "volume": mesh.volume if mesh.is_volume else 0,
            "surface_area": mesh.area,
            "has_degenerate_faces": len(mesh.faces) > 0 and np.any(np.abs(mesh.face_areas) < 1e-10),
            "euler_number": mesh.euler_number,
        }
    
    @staticmethod
    def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        mesh = MeshOptimizer.remove_duplicate_vertices(mesh)
        mesh = MeshOptimizer.remove_degenerate_faces(mesh)
        try:
            mesh.merge_vertices()
        except Exception:
            pass
        return mesh
    
    @staticmethod
    def validate_and_repair(mesh: trimesh.Trimesh, auto_repair: bool = True) -> Tuple[trimesh.Trimesh, dict]:
        quality_before = MeshValidator.check_mesh_quality(mesh)
        if auto_repair:
            mesh = MeshValidator.repair_mesh(mesh)
        quality_after = MeshValidator.check_mesh_quality(mesh)
        return mesh, {
            "before": quality_before,
            "after": quality_after,
            "repaired": auto_repair,
        }
