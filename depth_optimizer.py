"""
Advanced depth map optimization for high-quality 3D generation.
Includes multi-scale blending, edge preservation, and normal map generation.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List
from scipy.ndimage import gaussian_filter, laplace
from scipy.signal import medfilt2d


class DepthOptimizer:
    """Advanced depth processing with multiple optimization strategies."""
    
    @staticmethod
    def bilateral_denoise(depth: np.ndarray, diameter: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """Edge-preserving bilateral filtering."""
        depth_uint8 = np.clip(depth, 0, 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(depth_uint8, diameter, sigma_color, sigma_space)
        return filtered.astype(np.float32)
    
    @staticmethod
    def laplacian_sharpen(depth: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Sharpen details using Laplacian operator."""
        laplacian = laplace(depth)
        sharpened = depth - laplacian * strength
        return np.clip(sharpened, 0, 255)
    
    @staticmethod
    def adaptive_clahe(depth: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """Adaptive histogram equalization with edge preservation."""
        depth_uint8 = np.clip(depth, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        equalized = clahe.apply(depth_uint8)
        return equalized.astype(np.float32)
    
    @staticmethod
    def compute_gradients(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient magnitude and direction."""
        grad_y, grad_x = np.gradient(depth)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        return magnitude, direction
    
    @staticmethod
    def edge_aware_smooth(depth: np.ndarray, sigma: float = 1.0, iterations: int = 3) -> np.ndarray:
        """Smooth depth while preserving edges using structure tensor."""
        result = depth.copy()
        
        for _ in range(iterations):
            # Compute gradients
            grad_y, grad_x = np.gradient(result)
            
            # Structure tensor components
            Ixx = gaussian_filter(grad_x * grad_x, sigma)
            Iyy = gaussian_filter(grad_y * grad_y, sigma)
            Ixy = gaussian_filter(grad_x * grad_y, sigma)
            
            # Compute eigenvalues for edge detection
            trace = Ixx + Iyy
            det = Ixx * Iyy - Ixy * Ixy
            
            # Avoid division by zero
            lambda1 = (trace + np.sqrt(np.maximum(trace**2 - 4*det, 0))) / 2
            lambda2 = (trace - np.sqrt(np.maximum(trace**2 - 4*det, 0))) / 2
            
            # Edge weight: high where lambda1 >> lambda2 (edge), low elsewhere
            edge_weight = 1.0 / (1.0 + lambda1 / (lambda2 + 1e-6))
            
            # Apply weighted gaussian filter
            blurred = gaussian_filter(result, sigma)
            result = result * edge_weight + blurred * (1 - edge_weight)
        
        return np.clip(result, 0, 255)
    
    @staticmethod
    def multi_scale_blend(depth_layers: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """Intelligently blend multiple depth scales using frequency analysis."""
        if not depth_layers:
            return np.zeros((1, 1), dtype=np.float32)
        
        if weights is None:
            # Compute weights based on frequency content
            weights = []
            for layer in depth_layers:
                # Compute Laplacian (high-pass filter) to measure detail
                detail = np.abs(laplace(layer))
                detail_score = np.mean(detail)
                weights.append(detail_score + 0.1)  # Avoid zero weights
            
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        # Blend using computed weights
        result = np.zeros_like(depth_layers[0], dtype=np.float32)
        for layer, weight in zip(depth_layers, weights):
            result += layer.astype(np.float32) * weight
        
        return np.clip(result, 0, 255)
    
    @staticmethod
    def generate_normal_map(depth: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Generate normal map from depth for displacement enhancement."""
        # Compute gradients
        grad_y, grad_x = np.gradient(depth, axis=(0, 1))
        
        # Scale gradients
        grad_x = grad_x * scale
        grad_y = grad_y * scale
        
        # Create normal vectors (Sobel-like)
        # Normal = [-dx, -dy, 1] normalized
        normals = np.zeros((*depth.shape, 3), dtype=np.float32)
        normals[..., 0] = -grad_x
        normals[..., 1] = -grad_y
        normals[..., 2] = 1.0
        
        # Normalize
        magnitude = np.sqrt(np.sum(normals**2, axis=2, keepdims=True)) + 1e-6
        normals = normals / magnitude
        
        # Convert to RGB ([-1,1] -> [0,255])
        normal_map = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
        return normal_map
    
    @staticmethod
    def enhance_depth_contrast(depth: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Enhance depth contrast using sigmoid-like curve."""
        # Normalize to [-1, 1]
        mn, mx = depth.min(), depth.max()
        if mx > mn:
            normalized = 2 * (depth - mn) / (mx - mn) - 1
        else:
            normalized = depth
        
        # Apply sigmoid-like enhancement
        enhanced = 127.5 * (1 + np.tanh(normalized * strength)) 
        return np.clip(enhanced, 0, 255).astype(np.float32)
    
    @staticmethod
    def remove_outliers(depth: np.ndarray, percentile: float = 95) -> np.ndarray:
        """Remove depth outliers using percentile clipping."""
        p_low = np.percentile(depth, 2)
        p_high = np.percentile(depth, percentile)
        
        clipped = np.clip(depth, p_low, p_high)
        
        # Normalize back to 0-255
        if p_high > p_low:
            clipped = (clipped - p_low) / (p_high - p_low) * 255
        
        return clipped.astype(np.float32)
    
    @staticmethod
    def morphological_close(depth: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Morphological closing to fill small holes."""
        depth_uint8 = np.clip(depth, 0, 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(depth_uint8, cv2.MORPH_CLOSE, kernel)
        return closed.astype(np.float32)


class DepthPipeline:
    """Complete depth processing pipeline."""
    
    def __init__(self):
        self.optimizer = DepthOptimizer()
    
    def process(
        self,
        depth: np.ndarray,
        remove_outliers: bool = True,
        bilateral_denoise: bool = True,
        edge_aware_smooth: bool = True,
        clahe_enhance: bool = True,
        sharpen: bool = True,
        morph_close: bool = True,
    ) -> np.ndarray:
        """Apply complete processing pipeline."""
        result = depth.copy().astype(np.float32)
        
        if remove_outliers:
            result = self.optimizer.remove_outliers(result)
        
        if bilateral_denoise:
            result = self.optimizer.bilateral_denoise(result)
        
        if morph_close:
            result = self.optimizer.morphological_close(result)
        
        if edge_aware_smooth:
            result = self.optimizer.edge_aware_smooth(result, sigma=1.0, iterations=2)
        
        if clahe_enhance:
            result = self.optimizer.adaptive_clahe(result, clip_limit=2.0)
        
        if sharpen:
            result = self.optimizer.laplacian_sharpen(result, strength=0.3)
        
        return np.clip(result, 0, 255).astype(np.uint8)
