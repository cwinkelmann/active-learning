"""
Simplified point sampling with RGB contrast analysis.

Starting fresh with basic RGB contrast metrics between iguana and background regions.
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import json
from dataclasses import dataclass
import math

@dataclass
class SamplingConfig:
    """Configuration parameters for point sampling."""
    csv_path: str
    image_dir: str
    buffer_radius: int = 5
    patch_size: int = 32
    min_distance: int = 50
    object_size_pixels: int = 8
    max_images: Optional[int] = None
    visualize_first_n: int = 3
    output_dir: str = '/mnt/user-data/outputs'


class RGBContrastSampler:
    """Simple sampler focused on RGB contrast analysis."""
    
    def __init__(self, config: SamplingConfig):
        """Initialize the sampler with configuration and load annotations."""
        self.config = config
        self.buffer_radius = config.buffer_radius
        self.patch_size = config.patch_size
        self.min_distance = config.min_distance
        self.object_size_pixels = config.object_size_pixels
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("INITIALIZING RGB CONTRAST SAMPLER")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  CSV path:           {config.csv_path}")
        print(f"  Image directory:    {config.image_dir}")
        print(f"  Buffer radius:      {config.buffer_radius} pixels")
        print(f"  Min distance:       {config.min_distance} pixels")
        print(f"  Object size:        {config.object_size_pixels} pixels")
        print(f"  Output directory:   {config.output_dir}")
        
        self.annotations_df = self._load_annotations(config.csv_path)
        self.image_dir = Path(config.image_dir)
        
        self.unique_images = self.annotations_df['image_path'].unique()
        if config.max_images:
            self.unique_images = self.unique_images[:config.max_images]
        
        print(f"\n✓ Loaded {len(self.annotations_df)} annotations")
        print(f"✓ Found {len(self.unique_images)} unique images to process")
        print("="*70 + "\n")
    
    def _load_annotations(self, csv_path: str) -> pd.DataFrame:
        """Load and standardize point annotations."""
        df = pd.read_csv(csv_path)
        
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'image' in col_lower or 'filename' in col_lower or 'file' in col_lower:
                column_mapping[col] = 'image_path'
            elif col_lower == 'x':
                column_mapping[col] = 'x'
            elif col_lower == 'y':
                column_mapping[col] = 'y'
        
        df = df.rename(columns=column_mapping)
        
        required_cols = ['image_path', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _create_circular_mask(self, center: Tuple[int, int], 
                             image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a circular mask around a center point."""
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        cx, cy = center
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = dist_from_center <= self.buffer_radius
        return mask

    def _to_grayscale_luminance(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB image or RGB sample array to grayscale luminance.
        Accepts HxWx3 or Nx3 arrays.
        """
        # Rec. 709 luminance
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        if rgb.ndim == 3:
            return (rgb * weights).sum(axis=-1)
        elif rgb.ndim == 2 and rgb.shape[1] == 3:
            return (rgb * weights).sum(axis=-1)
        else:
            raise ValueError("Unexpected shape for RGB -> grayscale conversion")

    def _bilinear_sample(self, image: np.ndarray, x: float, y: float) -> np.ndarray:
        """
        Bilinear sample RGB at floating-point (x, y).
        Returns a (3,) RGB vector clipped to image bounds.
        """
        H, W = image.shape[:2]
        if x < 0 or y < 0 or x > W - 1 or y > H - 1:
            # clamp to edge
            xi = int(np.clip(round(x), 0, W - 1))
            yi = int(np.clip(round(y), 0, H - 1))
            return image[yi, xi, :].astype(np.float32)

        x0 = int(math.floor(x));
        x1 = min(x0 + 1, W - 1)
        y0 = int(math.floor(y));
        y1 = min(y0 + 1, H - 1)

        dx = x - x0;
        dy = y - y0

        Ia = image[y0, x0, :].astype(np.float32)
        Ib = image[y0, x1, :].astype(np.float32)
        Ic = image[y1, x0, :].astype(np.float32)
        Id = image[y1, x1, :].astype(np.float32)

        top = Ia * (1 - dx) + Ib * dx
        bottom = Ic * (1 - dx) + Id * dx
        val = top * (1 - dy) + bottom * dy
        return val

    def _sample_line_profile(self, image: np.ndarray, center: Tuple[int, int],
                             angle_deg: float, max_len: float, step: float = 1.0) -> np.ndarray:
        """
        Sample RGB along a ray starting at 'center' (x, y) for 'max_len' pixels, every 'step'.
        Returns array of shape (N, 3) RGB.
        """
        cx, cy = float(center[0]), float(center[1])
        theta = math.radians(angle_deg)
        dx, dy = math.cos(theta), math.sin(theta)

        samples = []
        t = 0.0
        while t <= max_len:
            x = cx + dx * t
            y = cy + dy * t
            samples.append(self._bilinear_sample(image, x, y))
            t += step
        return np.stack(samples, axis=0)  # (N, 3)

    def _compute_radial_edge_contrast(self,
                                      image: np.ndarray,
                                      point: Tuple[int, int],
                                      object_radius: int,
                                      gap: int,
                                      bg_width: int,
                                      angles_deg: List[float]) -> Dict:
        """
        For each angle, compare inner window [0, object_radius] vs background
        window [object_radius + gap, object_radius + gap + bg_width] along a ray.

        Returns per-angle and aggregated stats in grayscale (plus per-channel if needed).
        """
        H, W = image.shape[:2]
        # Ensure RGB
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Max length: stop at image edge in worst direction, or a safe margin
        max_len = max(H, W)

        per_angle = []
        for a in angles_deg:
            prof_rgb = self._sample_line_profile(image, point, a, max_len=max_len, step=1.0)  # (N,3)
            prof_y = self._to_grayscale_luminance(prof_rgb)  # (N,)

            # Guard for short profiles near edges
            n = len(prof_y)
            i0 = 0
            i1 = min(object_radius, n)  # object window end
            j0 = min(object_radius + gap, n)  # background window start
            j1 = min(object_radius + gap + bg_width, n)  # background window end

            if i1 - i0 < 3 or j1 - j0 < 3:
                # Not enough pixels; skip this angle
                continue

            obj_vals = prof_y[i0:i1]
            bg_vals = prof_y[j0:j1]

            obj_mean = float(np.mean(obj_vals))
            bg_mean = float(np.mean(bg_vals))
            bg_std = float(np.std(bg_vals) + 1e-10)

            contrast = obj_mean - bg_mean  # grayscale contrast along this ray
            cnr = contrast / bg_std  # grayscale CNR along this ray

            per_angle.append({
                "angle": a,
                "obj_mean": obj_mean,
                "bg_mean": bg_mean,
                "bg_std": bg_std,
                "contrast": contrast,
                "cnr": cnr
            })

        if not per_angle:
            # Fallback empty result if all rays invalid
            return {
                "per_angle": [],
                "mean_abs_contrast": 0.0,
                "max_abs_contrast": 0.0,
                "mean_abs_cnr": 0.0,
                "max_abs_cnr": 0.0,
            }

        contrasts = np.array([abs(x["contrast"]) for x in per_angle], dtype=np.float32)
        cnrs = np.array([abs(x["cnr"]) for x in per_angle], dtype=np.float32)

        summary = {
            "per_angle": per_angle,
            "mean_abs_contrast": float(np.mean(contrasts)),
            "max_abs_contrast": float(np.max(contrasts)),
            "mean_abs_cnr": float(np.mean(cnrs)),
            "max_abs_cnr": float(np.max(cnrs)),
        }
        return summary

    def _assess_edge_detectability(self,
                                   image: np.ndarray,
                                   pos_point: Tuple[int, int],
                                   neg_points: np.ndarray,
                                   object_radius: int,
                                   gap: int,
                                   bg_width: int,
                                   angles_deg: List[float],
                                   max_negatives_for_null: int = 20) -> Dict:
        """
        Compute radial edge-contrast metric for the positive point and compare it
        to a null distribution from randomly selected negative points.
        """
        # Positive point metric
        pos_metrics = self._compute_radial_edge_contrast(
            image=image,
            point=pos_point,
            object_radius=object_radius,
            gap=gap,
            bg_width=bg_width,
            angles_deg=angles_deg
        )

        pos_score = pos_metrics["mean_abs_cnr"]  # pick a robust aggregate
        # Build null from negatives
        K = min(len(neg_points), max_negatives_for_null)
        null_scores = []
        for i in range(K):
            neg_pt = tuple(map(int, neg_points[i]))
            neg_m = self._compute_radial_edge_contrast(
                image=image,
                point=neg_pt,
                object_radius=object_radius,
                gap=gap,
                bg_width=bg_width,
                angles_deg=angles_deg
            )
            null_scores.append(neg_m["mean_abs_cnr"])
        null_scores = np.array(null_scores, dtype=np.float32) if null_scores else np.array([0.0], dtype=np.float32)

        null_mean = float(np.mean(null_scores))
        null_std = float(np.std(null_scores) + 1e-10)

        # z-score and percentile vs null
        z = (pos_score - null_mean) / null_std if null_std > 0 else (1.0 if pos_score > null_mean else -1.0)
        percentile = float((np.sum(null_scores <= pos_score) / len(null_scores)) * 100.0)

        # Decision rule:
        # detectable if either strong z or high percentile vs null distribution
        is_detectable = (z >= 2.0) or (percentile >= 95.0)

        # Normalize a soft score in [0,1] for convenience (sigmoid of z, clipped)
        soft = 1.0 / (1.0 + math.exp(-z))
        score = float(np.clip(soft, 0.0, 1.0))

        return {
            "pos": pos_metrics,
            "null_mean_abs_cnr": null_mean,
            "null_std_abs_cnr": null_std,
            "z_score": float(z),
            "percentile_vs_null": percentile,
            "edge_detectable": is_detectable,
            "edge_score": score
        }

    def _generate_negative_samples(self, image_shape: Tuple[int, int],
                                   positive_points: np.ndarray,
                                   n_samples: int) -> np.ndarray:
        """Generate random negative sample points."""
        height, width = image_shape
        negative_points = []
        max_attempts = n_samples * 100
        attempts = 0
        
        while len(negative_points) < n_samples and attempts < max_attempts:
            attempts += 1
            margin = self.buffer_radius + 5
            x = np.random.randint(margin, width - margin)
            y = np.random.randint(margin, height - margin)
            
            point = np.array([x, y])
            distances = np.linalg.norm(positive_points - point, axis=1)
            
            if np.all(distances >= self.min_distance):
                negative_points.append([x, y])
        
        if len(negative_points) < n_samples:
            print(f"  Warning: Generated {len(negative_points)}/{n_samples} negative samples")
        
        return np.array(negative_points)
    
    def _extract_crop(self, image: np.ndarray, center: Tuple[int, int], 
                     crop_size: int = 140) -> np.ndarray:
        """
        Extract a square crop around a point.
        
        Args:
            image: Input image
            center: (x, y) center coordinates
            crop_size: Size of crop in pixels
            
        Returns:
            Square crop of size (crop_size, crop_size)
        """
        x, y = center
        half_size = crop_size // 2
        
        # Calculate bounds with edge handling
        y_start = max(0, y - half_size)
        y_end = min(image.shape[0], y + half_size)
        x_start = max(0, x - half_size)
        x_end = min(image.shape[1], x + half_size)
        
        crop = image[y_start:y_end, x_start:x_end]
        
        # Pad if necessary to achieve crop_size
        if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
            pad_y_before = (crop_size - crop.shape[0]) // 2
            pad_y_after = crop_size - crop.shape[0] - pad_y_before
            pad_x_before = (crop_size - crop.shape[1]) // 2
            pad_x_after = crop_size - crop.shape[1] - pad_x_before
            
            crop = np.pad(crop, 
                         ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after), (0, 0)), 
                         mode='edge')
        
        return crop[:crop_size, :crop_size]
    
    def _sample_buffer_pixels(self, image: np.ndarray, 
                             point: Tuple[int, int]) -> np.ndarray:
        """Sample all pixels within buffer radius."""
        mask = self._create_circular_mask(point, image.shape[:2])
        pixels = image[mask]
        return pixels
    
    def _compute_rgb_contrast_metrics(self, iguana_pixels: np.ndarray, 
                                     background_pixels: np.ndarray) -> Dict:
        """
        Compute RGB contrast metrics.
        
        Args:
            iguana_pixels: RGB pixels from iguana region (N, 3)
            background_pixels: RGB pixels from background region (M, 3)
            
        Returns:
            Dictionary with contrast metrics
        """
        # Mean colors
        mean_iguana = np.mean(iguana_pixels, axis=0)  # [R, G, B]
        mean_background = np.mean(background_pixels, axis=0)  # [R, G, B]
        
        # Standard deviations
        std_iguana = np.std(iguana_pixels, axis=0)
        std_background = np.std(background_pixels, axis=0)
        
        # Per-channel contrast
        contrast_rgb = mean_iguana - mean_background  # [ΔR, ΔG, ΔB]
        
        # Euclidean distance in RGB space
        euclidean_distance = np.linalg.norm(contrast_rgb)
        
        # Per-channel CNR
        cnr_rgb = contrast_rgb / (std_background + 1e-10)
        
        # Overall metrics (grayscale equivalents)
        mean_gray_iguana = np.mean(mean_iguana)
        mean_gray_background = np.mean(mean_background)
        std_gray_iguana = np.mean(std_iguana)
        std_gray_background = np.mean(std_background)
        
        contrast_gray = mean_gray_iguana - mean_gray_background
        cnr_gray = contrast_gray / (std_gray_background + 1e-10)
        
        # Michelson contrast (per channel and overall)
        michelson_rgb = np.abs(mean_iguana - mean_background) / (mean_iguana + mean_background + 1e-10)
        michelson_gray = abs(contrast_gray) / (mean_gray_iguana + mean_gray_background + 1e-10)
        
        # Weber fraction
        weber_fraction = abs(contrast_gray) / (mean_gray_background + 1.0)
        
        return {
            # RGB channel means
            'mean_iguana_rgb': mean_iguana.tolist(),
            'mean_background_rgb': mean_background.tolist(),
            'mean_iguana_r': float(mean_iguana[0]),
            'mean_iguana_g': float(mean_iguana[1]),
            'mean_iguana_b': float(mean_iguana[2]),
            'mean_background_r': float(mean_background[0]),
            'mean_background_g': float(mean_background[1]),
            'mean_background_b': float(mean_background[2]),
            
            # RGB channel std
            'std_iguana_rgb': std_iguana.tolist(),
            'std_background_rgb': std_background.tolist(),
            
            # Per-channel contrast
            'contrast_r': float(contrast_rgb[0]),
            'contrast_g': float(contrast_rgb[1]),
            'contrast_b': float(contrast_rgb[2]),
            'contrast_rgb_vector': contrast_rgb.tolist(),
            
            # Euclidean distance
            'euclidean_distance': float(euclidean_distance),
            
            # Per-channel CNR
            'cnr_r': float(cnr_rgb[0]),
            'cnr_g': float(cnr_rgb[1]),
            'cnr_b': float(cnr_rgb[2]),
            'cnr_rgb_vector': cnr_rgb.tolist(),
            
            # Grayscale equivalents
            'mean_gray_iguana': float(mean_gray_iguana),
            'mean_gray_background': float(mean_gray_background),
            'contrast_gray': float(contrast_gray),
            'cnr_gray': float(cnr_gray),
            
            # Normalized metrics
            'michelson_r': float(michelson_rgb[0]),
            'michelson_g': float(michelson_rgb[1]),
            'michelson_b': float(michelson_rgb[2]),
            'michelson_gray': float(michelson_gray),
            'weber_fraction': float(weber_fraction)
        }
    
    def _get_image_path(self, img_name: str) -> Optional[Path]:
        """Find the full path to an image."""
        img_path = self.image_dir / Path(img_name).name
        if img_path.exists():
            return img_path
        
        img_path = Path(img_name)
        if img_path.exists():
            return img_path
        
        return None
    
    def process_image(self, img_name: str, visualize: bool = False) -> Optional[Dict]:
        """Process a single image with RGB contrast analysis."""
        img_path = self._get_image_path(img_name)
        if img_path is None:
            print(f"  ✗ Image not found: {img_name}")
            return None
        
        # Load image as RGB
        image = np.array(Image.open(img_path))



        # Ensure RGB format
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            # RGBA - drop alpha channel
            image = image[:, :, :3]

        # === New: Radial edge-contrast detectability ===
        # Set windows: object radius ~ half your object size; gap to avoid mixed pixels; bg window width
        object_radius = max(3, int(round(self.object_size_pixels / 2)))
        gap = max(2, self.buffer_radius // 2)  # small gap between object and background
        bg_width = max(6, self.buffer_radius)  # collect stable background stats
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]

        edge_results = None



        height, width = image.shape[:2]
        
        # Get annotations
        img_annotations = self.annotations_df[
            self.annotations_df['image_path'].str.contains(Path(img_name).name)
        ]
        
        if len(img_annotations) == 0:
            print(f"  ✗ No annotations found: {img_name}")
            return None
        
        positive_points = img_annotations[['x', 'y']].values
        n_positive = len(positive_points)
        
        # Generate negative samples
        negative_points = self._generate_negative_samples(
            (height, width), 
            positive_points, 
            n_positive
        )

        if len(positive_points) > 0 and len(negative_points) > 0:
            # use the first iguana point (consistent with your crops/visuals)
            pos_pt = tuple(map(int, positive_points[0]))
            edge_results = self._assess_edge_detectability(
                image=image,
                pos_point=pos_pt,
                neg_points=negative_points,
                object_radius=object_radius,
                gap=gap,
                bg_width=bg_width,
                angles_deg=angles_deg,
                max_negatives_for_null=20
            )
        
        # Sample RGB pixels
        positive_pixels_list = []
        for point in positive_points:
            pixels = self._sample_buffer_pixels(image, tuple(point))
            positive_pixels_list.append(pixels)
        
        negative_pixels_list = []
        for point in negative_points:
            pixels = self._sample_buffer_pixels(image, tuple(point))
            negative_pixels_list.append(pixels)
        
        # Concatenate all pixels
        positive_pixels_all = np.concatenate(positive_pixels_list, axis=0)  # (N, 3)
        negative_pixels_all = np.concatenate(negative_pixels_list, axis=0)  # (M, 3)
        
        # Compute RGB contrast metrics
        rgb_metrics = self._compute_rgb_contrast_metrics(
            positive_pixels_all,
            negative_pixels_all
        )
        
        # Assess detectability
        detectability = self._assess_detectability(rgb_metrics)
        
        # Compile results
        results = {
            'image_path': str(img_path),
            'image_name': img_name,
            'n_positive_points': n_positive,
            'n_negative_points': len(negative_points),
            'positive_points': positive_points.tolist(),
            'negative_points': negative_points.tolist(),
            'rgb_metrics': rgb_metrics,
            'detectability': detectability,
            'edge_detectability': edge_results  # <-- NEW
        }
        
        # Save crop patches (first iguana and first negative sample)
        if len(positive_points) > 0 and len(negative_points) > 0:
            self._save_crop_patches(image, positive_points[0], negative_points[0], results)
        
        # Visualization
        if visualize:
            self._visualize_rgb_contrast(image, positive_points, negative_points, results)
        
        return results
    
    def _save_crop_patches(self, image: np.ndarray, 
                          iguana_point: np.ndarray, 
                          negative_point: np.ndarray,
                          results: Dict):
        """
        Save 140px crops of iguana and negative sample.
        
        Args:
            image: Full image
            iguana_point: (x, y) coordinates of iguana
            negative_point: (x, y) coordinates of negative sample
            results: Results dictionary with image info
        """
        crop_size = 140
        
        # Extract crops
        iguana_crop = self._extract_crop(image, tuple(iguana_point), crop_size)
        negative_crop = self._extract_crop(image, tuple(negative_point), crop_size)
        
        # Create output directory for crops
        crops_dir = self.output_dir / 'crops'
        crops_dir.mkdir(exist_ok=True)
        
        # Save crops
        image_stem = Path(results['image_name']).stem
        
        iguana_path = crops_dir / f"{image_stem}_iguana_crop.png"
        negative_path = crops_dir / f"{image_stem}_negative_crop.png"
        
        Image.fromarray(iguana_crop).save(iguana_path)
        Image.fromarray(negative_crop).save(negative_path)
        
        # Add crop paths to results
        results['iguana_crop_path'] = str(iguana_path)
        results['negative_crop_path'] = str(negative_path)
    
    def _visualize_rgb_contrast(self, image: np.ndarray,
                                positive_points: np.ndarray,
                                negative_points: np.ndarray,
                                results: Dict):
        """Create visualization of RGB contrast analysis."""
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        rgb = results['rgb_metrics']
        detect = results['detectability']
        
        # Row 1, Col 1: Image with sampled points
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        
        # Draw circles first (as backgrounds)
        for point in positive_points:
            circle = plt.Circle(point, self.buffer_radius, 
                              color='red', fill=False, linewidth=2, alpha=0.7)
            ax1.add_patch(circle)
        
        for point in negative_points:
            circle = plt.Circle(point, self.buffer_radius, 
                              color='blue', fill=False, linewidth=2, alpha=0.7)
            ax1.add_patch(circle)
        
        # Then add markers on top (smaller, empty)
        ax1.scatter(positive_points[:, 0], positive_points[:, 1], 
                   c='none', s=50, marker='o', edgecolor='red', linewidths=2, label='Iguana')
        ax1.scatter(negative_points[:, 0], negative_points[:, 1], 
                   c='none', s=50, marker='o', edgecolor='blue', linewidths=2, label='Background')
        
        ax1.set_title('Sampled Regions', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.axis('off')
        
        # Row 1, Col 2: RGB mean colors
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Create color patches
        iguana_color = np.array(rgb['mean_iguana_rgb']) / 255.0
        background_color = np.array(rgb['mean_background_rgb']) / 255.0
        
        colors = np.array([[iguana_color, background_color]])
        ax2.imshow(colors, aspect='auto')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Iguana', 'Background'])
        ax2.set_yticks([])
        ax2.set_title('Mean RGB Colors', fontsize=14, fontweight='bold')
        
        # Add RGB values as text
        ax2.text(0, 0, f'R: {rgb["mean_iguana_r"]:.1f}\nG: {rgb["mean_iguana_g"]:.1f}\nB: {rgb["mean_iguana_b"]:.1f}',
                ha='center', va='center', fontsize=10, color='white' if np.mean(iguana_color) < 0.5 else 'black',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        ax2.text(1, 0, f'R: {rgb["mean_background_r"]:.1f}\nG: {rgb["mean_background_g"]:.1f}\nB: {rgb["mean_background_b"]:.1f}',
                ha='center', va='center', fontsize=10, color='white' if np.mean(background_color) < 0.5 else 'black',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        
        # Row 1, Col 3: Detectability conclusion
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        
        status = "DETECTABLE ✓" if detect['is_detectable'] else "NOT DETECTABLE ✗"
        
        conclusion_text = f"""
        {status}
        
        Score: {detect['detectability_score']:.3f}
        Confidence: {detect['confidence'].upper()}
        
        ────────────────────
        Metrics Passed:
        {chr(10).join(['  ✓ ' + m for m in detect['passed_metrics']])}
        
        Metrics Failed:
        {chr(10).join(['  ✗ ' + m for m in detect['failed_metrics']])}
        """
        
        bbox_color = 'lightgreen' if detect['is_detectable'] else 'lightcoral'
        ax3.text(0.5, 0.5, conclusion_text, fontsize=10, 
                family='monospace', verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.4, pad=1),
                transform=ax3.transAxes)
        
        # Row 2, Col 1: Iguana crop
        ax4 = fig.add_subplot(gs[1, 0])
        if len(positive_points) > 0:
            iguana_crop = self._extract_crop(image, tuple(positive_points[0]), 140)
            ax4.imshow(iguana_crop)
            ax4.set_title('Iguana Crop (140px)', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # Row 2, Col 2: Negative crop
        ax5 = fig.add_subplot(gs[1, 1])
        if len(negative_points) > 0:
            negative_crop = self._extract_crop(image, tuple(negative_points[0]), 140)
            ax5.imshow(negative_crop)
            ax5.set_title('Background Crop (140px)', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # Row 2, Col 3: Per-channel contrast
        ax6 = fig.add_subplot(gs[1, 2])
        channels = ['R', 'G', 'B']
        contrasts = [rgb['contrast_r'], rgb['contrast_g'], rgb['contrast_b']]
        colors_bar = ['red', 'green', 'blue']
        
        bars = ax6.bar(channels, contrasts, color=colors_bar, alpha=0.7, edgecolor='black')
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax6.set_ylabel('Contrast (Iguana - Background)', fontsize=11)
        ax6.set_title('Per-Channel Contrast', fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, contrasts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # Row 3, Col 1: Per-channel CNR
        ax7 = fig.add_subplot(gs[2, 0])
        cnr_values = [rgb['cnr_r'], rgb['cnr_g'], rgb['cnr_b']]
        
        bars = ax7.bar(channels, cnr_values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax7.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (1.0)')
        ax7.set_ylabel('CNR', fontsize=11)
        ax7.set_title('Per-Channel CNR', fontsize=14, fontweight='bold')
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, cnr_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # Row 3, Col 2: Michelson contrast per channel
        ax8 = fig.add_subplot(gs[2, 1])
        michelson_values = [rgb['michelson_r'], rgb['michelson_g'], rgb['michelson_b']]
        
        bars = ax8.bar(channels, michelson_values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax8.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (0.05)')
        ax8.set_ylabel('Michelson Contrast', fontsize=11)
        ax8.set_title('Per-Channel Michelson Contrast', fontsize=14, fontweight='bold')
        ax8.legend()
        ax8.grid(axis='y', alpha=0.3)
        ax8.set_ylim([0, max(michelson_values) * 1.2 if max(michelson_values) > 0 else 1])
        
        # Add value labels
        for bar, val in zip(bars, michelson_values):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        # Row 3, Col 3: Empty (reserved for future use)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Row 4: Metrics table spanning all columns
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')
        
        thresholds = detect.get('thresholds', {})
        
        metrics_text = f"""
        RGB CONTRAST METRICS & THRESHOLDS
        ════════════════════════════════════════════════════════════════════════════════════════
        
        Metric                      Value          Threshold      Status
        ────────────────────────────────────────────────────────────────────────────────────────
        Euclidean Distance:         {rgb['euclidean_distance']:6.2f}         ≥ {thresholds.get('euclidean', 10):4.1f}        {'✓ PASS' if detect['euclidean_ok'] else '✗ FAIL'}
        Absolute Contrast:          {abs(rgb['contrast_gray']):6.2f}         ≥ {thresholds.get('abs_contrast', 5):4.1f}        {'✓ PASS' if detect.get('abs_contrast_ok', False) else '✗ FAIL'}
        Grayscale CNR:              {rgb['cnr_gray']:6.3f}         ≥ {thresholds.get('cnr', 1):4.1f}        {'✓ PASS' if detect['cnr_ok'] else '✗ FAIL'}
        Michelson (gray):           {rgb['michelson_gray']:6.3f}         ≥ {thresholds.get('michelson', 0.05):4.2f}      {'✓ PASS' if detect['michelson_ok'] else '✗ FAIL'}
        Weber Fraction:             {rgb['weber_fraction']:6.3f}         ≥ {thresholds.get('weber', 0.02):4.2f}      {'✓ PASS' if detect['weber_ok'] else '✗ FAIL'}
        Any Channel Contrast > 5:   {max(abs(rgb['contrast_r']), abs(rgb['contrast_g']), abs(rgb['contrast_b'])):6.2f}         ≥ {thresholds.get('abs_contrast', 5):4.1f}        {'✓ PASS' if detect.get('any_channel_contrast_ok', False) else '✗ FAIL'}
        Any Channel CNR > 1:        {max(rgb['cnr_r'], rgb['cnr_g'], rgb['cnr_b']):6.3f}         ≥ {thresholds.get('cnr', 1):4.1f}        {'✓ PASS' if detect['any_channel_cnr_ok'] else '✗ FAIL'}
        
        ────────────────────────────────────────────────────────────────────────────────────────
        Per-Channel Values:         R: {rgb['contrast_r']:6.2f}    G: {rgb['contrast_g']:6.2f}    B: {rgb['contrast_b']:6.2f}
        Per-Channel CNR:            R: {rgb['cnr_r']:6.3f}    G: {rgb['cnr_g']:6.3f}    B: {rgb['cnr_b']:6.3f}
        """
        
        ax10.text(0.05, 0.5, metrics_text, fontsize=10, 
                family='monospace', verticalalignment='center')
        
        plt.suptitle(f'RGB Contrast Analysis: {Path(results["image_name"]).stem}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save
        output_filename = Path(results['image_name']).stem + '_rgb_contrast.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {output_filename}")
        plt.close()
        
        # Create color patches
        iguana_color = np.array(rgb['mean_iguana_rgb']) / 255.0
        background_color = np.array(rgb['mean_background_rgb']) / 255.0
        
        colors = np.array([[iguana_color, background_color]])
        ax2.imshow(colors, aspect='auto')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Iguana', 'Background'])
        ax2.set_yticks([])
        ax2.set_title('Mean RGB Colors', fontsize=14, fontweight='bold')
        
        # Add RGB values as text
        ax2.text(0, 0, f'R: {rgb["mean_iguana_r"]:.1f}\nG: {rgb["mean_iguana_g"]:.1f}\nB: {rgb["mean_iguana_b"]:.1f}',
                ha='center', va='center', fontsize=10, color='white' if np.mean(iguana_color) < 0.5 else 'black',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        ax2.text(1, 0, f'R: {rgb["mean_background_r"]:.1f}\nG: {rgb["mean_background_g"]:.1f}\nB: {rgb["mean_background_b"]:.1f}',
                ha='center', va='center', fontsize=10, color='white' if np.mean(background_color) < 0.5 else 'black',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.3))
        
        # Row 1, Col 3: Detectability conclusion
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        status = "DETECTABLE ✓" if detect['is_detectable'] else "NOT DETECTABLE ✗"

        conclusion_text = f"""
        {status}
        
        Score: {detect['detectability_score']:.3f}
        Confidence: {detect['confidence'].upper()}
        
        ────────────────────
        Metrics Passed:
        {chr(10).join(['  ✓ ' + m for m in detect['passed_metrics']])}
        
        Metrics Failed:
        {chr(10).join(['  ✗ ' + m for m in detect['failed_metrics']])}
        """

        bbox_color = 'lightgreen' if detect['is_detectable'] else 'lightcoral'
        ax3.text(0.5, 0.5, conclusion_text, fontsize=10,
                family='monospace', verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.4, pad=1),
                transform=ax3.transAxes)

        plt.suptitle(f'RGB Contrast Analysis: {Path(results["image_name"]).stem}',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save
        output_filename = Path(results['image_name']).stem + '_rgb_contrast.png'
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {output_filename}")
        plt.close()

    def run(self) -> pd.DataFrame:
        """Run the complete analysis pipeline."""
        print("STARTING RGB CONTRAST ANALYSIS")
        print("="*70)

        all_results = []

        for i, img_name in enumerate(self.unique_images):
            print(f"\n[{i+1}/{len(self.unique_images)}] {img_name}")

            visualize = (i < self.config.visualize_first_n)
            results = self.process_image(img_name, visualize=visualize)

            if results:
                all_results.append(results)
                rgb = results['rgb_metrics']
                detect = results['detectability']
                print(f"  Euclidean Distance: {rgb['euclidean_distance']:.2f}")
                print(f"  Gray CNR: {rgb['cnr_gray']:.3f} | Michelson: {rgb['michelson_gray']:.3f}")
                print(f"  Detectable: {'YES ✓' if detect['is_detectable'] else 'NO ✗'} "
                      f"(score: {detect['detectability_score']:.3f}, {detect['confidence']})")

        # Save results
        results_file = self.output_dir / 'rgb_contrast_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved detailed results: {results_file}")

        # Create summary
        summary_df = self._create_summary(all_results)
        self._print_statistics(summary_df)

        # Create summary plot
        self._create_summary_plot(summary_df)

        return summary_df

    def _create_summary_plot(self, summary_df: pd.DataFrame):
        """Create comprehensive summary plot for the entire dataset."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Color coding by detectability
        colors = ['green' if d else 'red' for d in summary_df['is_detectable']]

        # Plot 1: Euclidean Distance
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(range(len(summary_df)), summary_df['euclidean_distance'],
                   c=colors, s=100, alpha=0.6, edgecolor='black')
        ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (10)')
        ax1.set_xlabel('Image Index', fontsize=11)
        ax1.set_ylabel('Euclidean Distance', fontsize=11)
        ax1.set_title('RGB Euclidean Distance', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gray CNR
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(range(len(summary_df)), summary_df['cnr_gray'],
                   c=colors, s=100, alpha=0.6, edgecolor='black')
        ax2.axhline(y=1, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (1.0)')
        ax2.set_xlabel('Image Index', fontsize=11)
        ax2.set_ylabel('CNR (Grayscale)', fontsize=11)
        ax2.set_title('Grayscale CNR', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Michelson Contrast
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(range(len(summary_df)), summary_df['michelson_gray'],
                   c=colors, s=100, alpha=0.6, edgecolor='black')
        ax3.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.05)')
        ax3.set_xlabel('Image Index', fontsize=11)
        ax3.set_ylabel('Michelson Contrast', fontsize=11)
        ax3.set_title('Michelson Contrast (Grayscale)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Per-channel CNR comparison
        ax4 = fig.add_subplot(gs[1, 0])
        x = np.arange(len(summary_df))
        width = 0.25
        ax4.bar(x - width, summary_df['cnr_r'], width, label='Red', color='red', alpha=0.7)
        ax4.bar(x, summary_df['cnr_g'], width, label='Green', color='green', alpha=0.7)
        ax4.bar(x + width, summary_df['cnr_b'], width, label='Blue', color='blue', alpha=0.7)
        ax4.axhline(y=1, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Image Index', fontsize=11)
        ax4.set_ylabel('CNR', fontsize=11)
        ax4.set_title('Per-Channel CNR Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Plot 5: Detectability score distribution
        ax5 = fig.add_subplot(gs[1, 1])
        bars = ax5.bar(range(len(summary_df)), summary_df['detectability_score'],
                      color=colors, alpha=0.7, edgecolor='black')
        ax5.axhline(y=0.4, color='black', linestyle='-', linewidth=2, label='Decision Threshold (0.4)')
        ax5.set_xlabel('Image Index', fontsize=11)
        ax5.set_ylabel('Detectability Score', fontsize=11)
        ax5.set_title('Detectability Score', fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 1.0])
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # Plot 6: Detectability pie chart
        ax6 = fig.add_subplot(gs[1, 2])
        detectable_count = summary_df['is_detectable'].sum()
        not_detectable_count = len(summary_df) - detectable_count

        if detectable_count + not_detectable_count > 0:
            ax6.pie([detectable_count, not_detectable_count],
                   labels=['Detectable', 'Not Detectable'],
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax6.set_title('Detectability Distribution', fontsize=12, fontweight='bold')

        # Plot 7: Confidence breakdown
        ax7 = fig.add_subplot(gs[2, 0])
        confidence_counts = summary_df['confidence'].value_counts()
        colors_conf = {'high': 'darkgreen', 'medium': 'orange', 'low': 'darkred'}
        bar_colors = [colors_conf.get(c, 'gray') for c in confidence_counts.index]

        ax7.bar(confidence_counts.index, confidence_counts.values,
               color=bar_colors, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Confidence Level', fontsize=11)
        ax7.set_ylabel('Count', fontsize=11)
        ax7.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (idx, val) in enumerate(confidence_counts.items()):
            ax7.text(i, val, str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 8: Contrast vs CNR scatter
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(summary_df['contrast_gray'], summary_df['cnr_gray'],
                   c=colors, s=100, alpha=0.6, edgecolor='black')
        ax8.axhline(y=1, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax8.set_xlabel('Grayscale Contrast', fontsize=11)
        ax8.set_ylabel('Grayscale CNR', fontsize=11)
        ax8.set_title('Contrast vs CNR', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)

        # Plot 9: Summary statistics table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        n_detectable = summary_df['is_detectable'].sum()
        pct_detectable = 100 * n_detectable / len(summary_df) if len(summary_df) > 0 else 0

        stats_text = f"""
        DATASET SUMMARY
        ═══════════════════════════
        
        Total Images:      {len(summary_df)}
        Detectable:        {n_detectable} ({pct_detectable:.1f}%)
        Not Detectable:    {len(summary_df) - n_detectable}
        
        Mean Euclidean:    {summary_df['euclidean_distance'].mean():.2f}
        Mean CNR:          {summary_df['cnr_gray'].mean():.3f}
        Mean Michelson:    {summary_df['michelson_gray'].mean():.3f}
        Mean Weber:        {summary_df['weber_fraction'].mean():.3f}
        
        ───────────────────────────
        Confidence:
          High:    {(summary_df['confidence'] == 'high').sum()}
          Medium:  {(summary_df['confidence'] == 'medium').sum()}
          Low:     {(summary_df['confidence'] == 'low').sum()}
        """

        ax9.text(0.1, 0.5, stats_text, fontsize=11,
                family='monospace', verticalalignment='center')

        # Overall title
        dataset_name = Path(self.config.image_dir).parent.name
        plt.suptitle(f'RGB Contrast Summary: {dataset_name}',
                    fontsize=18, fontweight='bold', y=0.98)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / 'dataset_summary_plot.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved summary plot: {output_path}")
        plt.close()

    def _create_summary(self, all_results: List[Dict]) -> pd.DataFrame:
        """Create summary DataFrame."""
        summary_data = []

        for r in all_results:
            rgb = r['rgb_metrics']
            detect = r['detectability']
            edge = r.get('edge_detectability') or {}
            pos = (edge or {}).get('pos') or {}
            summary_data.append({
                'image': r['image_name'],
                'euclidean_distance': rgb['euclidean_distance'],
                'contrast_gray': rgb['contrast_gray'],
                'cnr_gray': rgb['cnr_gray'],
                'michelson_gray': rgb['michelson_gray'],
                'weber_fraction': rgb['weber_fraction'],
                'contrast_r': rgb['contrast_r'],
                'contrast_g': rgb['contrast_g'],
                'contrast_b': rgb['contrast_b'],
                'cnr_r': rgb['cnr_r'],
                'cnr_g': rgb['cnr_g'],
                'cnr_b': rgb['cnr_b'],
                'is_detectable': detect['is_detectable'],
                'detectability_score': detect['detectability_score'],
                'confidence': detect['confidence'],
                'n_positive': r['n_positive_points'],
                # NEW columns
                'edge_mean_abs_cnr': pos.get('mean_abs_cnr', np.nan),
                'edge_max_abs_cnr': pos.get('max_abs_cnr', np.nan),
                'edge_z': edge.get('z_score', np.nan),
                'edge_percentile': edge.get('percentile_vs_null', np.nan),
                'edge_detectable': edge.get('edge_detectable', False),
                'edge_score': edge.get('edge_score', np.nan),
            })

        summary_df = pd.DataFrame(summary_data)

        summary_file = self.output_dir / 'rgb_contrast_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Saved summary: {summary_file}")

        return summary_df

    def _assess_detectability(self, rgb_metrics: Dict) -> Dict:
        """
        Assess if object is detectable based on RGB contrast metrics.

        Args:
            rgb_metrics: Dictionary with RGB contrast metrics

        Returns:
            Dictionary with detectability assessment
        """
        # More permissive thresholds for detectability
        EUCLIDEAN_THRESHOLD = 10.0  # Lower threshold - any visible difference
        CNR_THRESHOLD = 1.0  # Lower threshold - even low CNR can be detectable
        MICHELSON_THRESHOLD = 0.05  # Lower threshold
        WEBER_THRESHOLD = 0.02  # Lower threshold
        ABSOLUTE_CONTRAST_THRESHOLD = 5.0  # Just 5 gray levels difference

        # Individual metric checks
        euclidean_ok = rgb_metrics['euclidean_distance'] >= EUCLIDEAN_THRESHOLD
        cnr_ok = rgb_metrics['cnr_gray'] >= CNR_THRESHOLD
        michelson_ok = rgb_metrics['michelson_gray'] >= MICHELSON_THRESHOLD
        weber_ok = rgb_metrics['weber_fraction'] >= WEBER_THRESHOLD

        # Absolute contrast - is there ANY visible difference?
        abs_contrast = abs(rgb_metrics['contrast_gray'])
        abs_contrast_ok = abs_contrast >= ABSOLUTE_CONTRAST_THRESHOLD

        # Check if any channel has contrast
        any_channel_contrast_ok = (abs(rgb_metrics['contrast_r']) >= ABSOLUTE_CONTRAST_THRESHOLD or
                                   abs(rgb_metrics['contrast_g']) >= ABSOLUTE_CONTRAST_THRESHOLD or
                                   abs(rgb_metrics['contrast_b']) >= ABSOLUTE_CONTRAST_THRESHOLD)

        # Check if any channel has CNR
        any_channel_cnr_ok = (rgb_metrics['cnr_r'] >= CNR_THRESHOLD or
                              rgb_metrics['cnr_g'] >= CNR_THRESHOLD or
                              rgb_metrics['cnr_b'] >= CNR_THRESHOLD)

        # Scoring system (weighted voting)
        # If object is clearly visible, multiple metrics should pass
        scores = {
            'euclidean': (euclidean_ok, 1.5),
            'cnr_gray': (cnr_ok, 1.5),
            'michelson': (michelson_ok, 1.0),
            'weber': (weber_ok, 1.0),
            'abs_contrast': (abs_contrast_ok, 2.0),  # High weight - most important
            'any_channel_contrast': (any_channel_contrast_ok, 1.5),
            'any_channel_cnr': (any_channel_cnr_ok, 1.0)
        }

        total_weight = sum(weight for _, weight in scores.values())
        positive_weight = sum(weight for passed, weight in scores.values() if passed)
        detectability_score = positive_weight / total_weight

        # More permissive final decision - if score > 0.4, it's detectable
        is_detectable = detectability_score >= 0.4

        # Confidence level
        if detectability_score >= 0.7:
            confidence = 'high'
        elif detectability_score >= 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Passed metrics
        passed = [name for name, (passed, _) in scores.items() if passed]
        failed = [name for name, (passed, _) in scores.items() if not passed]

        return {
            'is_detectable': is_detectable,
            'detectability_score': float(detectability_score),
            'confidence': confidence,
            'euclidean_ok': euclidean_ok,
            'cnr_ok': cnr_ok,
            'michelson_ok': michelson_ok,
            'weber_ok': weber_ok,
            'abs_contrast_ok': abs_contrast_ok,
            'any_channel_contrast_ok': any_channel_contrast_ok,
            'any_channel_cnr_ok': any_channel_cnr_ok,
            'passed_metrics': passed,
            'failed_metrics': failed,
            'thresholds': {
                'euclidean': EUCLIDEAN_THRESHOLD,
                'cnr': CNR_THRESHOLD,
                'michelson': MICHELSON_THRESHOLD,
                'weber': WEBER_THRESHOLD,
                'abs_contrast': ABSOLUTE_CONTRAST_THRESHOLD
            },
            'summary': f"Score: {detectability_score:.2f} | Passed: {', '.join(passed) if passed else 'none'}"
        }

    def _print_statistics(self, summary_df: pd.DataFrame):
        """Print overall statistics."""
        print("\n" + "="*70)
        print("OVERALL RGB CONTRAST STATISTICS")
        print("="*70)

        print("\nOVERALL METRICS:")
        print(f"  Mean Euclidean Dist:   {summary_df['euclidean_distance'].mean():.2f} ± {summary_df['euclidean_distance'].std():.2f}")
        print(f"  Mean Gray Contrast:    {summary_df['contrast_gray'].mean():.2f} ± {summary_df['contrast_gray'].std():.2f}")
        print(f"  Mean Gray CNR:         {summary_df['cnr_gray'].mean():.3f} ± {summary_df['cnr_gray'].std():.3f}")
        print(f"  Mean Michelson:        {summary_df['michelson_gray'].mean():.3f} ± {summary_df['michelson_gray'].std():.3f}")
        print(f"  Mean Weber:            {summary_df['weber_fraction'].mean():.3f} ± {summary_df['weber_fraction'].std():.3f}")

        print("\nPER-CHANNEL CONTRAST:")
        print(f"  Mean Red:              {summary_df['contrast_r'].mean():.2f} ± {summary_df['contrast_r'].std():.2f}")
        print(f"  Mean Green:            {summary_df['contrast_g'].mean():.2f} ± {summary_df['contrast_g'].std():.2f}")
        print(f"  Mean Blue:             {summary_df['contrast_b'].mean():.2f} ± {summary_df['contrast_b'].std():.2f}")

        print("\nPER-CHANNEL CNR:")
        print(f"  Mean Red:              {summary_df['cnr_r'].mean():.3f} ± {summary_df['cnr_r'].std():.3f}")
        print(f"  Mean Green:            {summary_df['cnr_g'].mean():.3f} ± {summary_df['cnr_g'].std():.3f}")
        print(f"  Mean Blue:             {summary_df['cnr_b'].mean():.3f} ± {summary_df['cnr_b'].std():.3f}")

        print("\nDETECTABILITY:")
        n_detectable = summary_df['is_detectable'].sum()
        pct_detectable = 100 * summary_df['is_detectable'].mean()
        print(f"  Detectable images:     {n_detectable}/{len(summary_df)} ({pct_detectable:.1f}%)")
        print(f"  Mean Detect. Score:    {summary_df['detectability_score'].mean():.3f}")

        for conf in ['high', 'medium', 'low']:
            count = (summary_df['confidence'] == conf).sum()
            pct = 100 * count / len(summary_df) if len(summary_df) > 0 else 0
            print(f"  {conf.capitalize()} confidence:      {count} ({pct:.1f}%)")

        print(f"\nTOTAL ANNOTATIONS:       {summary_df['n_positive'].sum()}")
        print("="*70 + "\n")


def main():
    """Main entry point."""
    base_path_floreana = Path('/raid/cwinkelmann/training_data/iguana/2025_10_11/Floreana_detection/train')
    base_path_floreana = Path('/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_s_detection/train')
    # base_path_floreana = Path('/raid/cwinkelmann/training_data/iguana/2025_10_11/Fernandina_m_detection/train')


    config = SamplingConfig(
        csv_path=base_path_floreana / 'herdnet_format_512_0_crops.csv',
        image_dir=base_path_floreana / 'crops_512_numNone_overlap0',
        output_dir=base_path_floreana / 'outputs',
        buffer_radius=5,
        patch_size=32,
        min_distance=50,
        object_size_pixels=8,
        max_images=30,
        visualize_first_n=15,

    )
    
    sampler = RGBContrastSampler(config)
    summary_df = sampler.run()
    
    print("\n" + "="*70)
    print("✓ RGB CONTRAST ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults in: {config.output_dir}")
    print("  • rgb_contrast_results.json")
    print("  • rgb_contrast_summary.csv")
    print("  • dataset_summary_plot.png")
    print("  • *_rgb_contrast.png")
    print("  • crops/*.png  (140px iguana and background crops)")
    print()


if __name__ == "__main__":
    main()