import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
import argparse

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: List[float]

@dataclass
class Frame:
    id: int
    rig_id: int
    quaternion: List[float]  # [QW, QX, QY, QZ]
    translation: List[float]  # [TX, TY, TZ]
    data_ids: List[Tuple[int, int, int]]  # (SENSOR_TYPE, SENSOR_ID, DATA_ID)

@dataclass
class Image:
    id: int
    quaternion: List[float]  # [QW, QX, QY, QZ]
    translation: List[float]  # [TX, TY, TZ]
    camera_id: int
    name: str
    points2d: List[Tuple[float, float, int]]  # (X, Y, POINT3D_ID)

@dataclass
class Point3D:
    id: int
    xyz: List[float]  # [X, Y, Z]
    rgb: List[int]  # [R, G, B]
    error: float
    track: List[Tuple[int, int]]  # (IMAGE_ID, POINT2D_IDX)

@dataclass
class Rig:
    id: int
    num_sensors: int
    ref_sensor_type: int
    ref_sensor_id: int
    sensors: List[Tuple]

class SFMDataParser:
    """Parser para archivos de datos SFM en formato COLMAP"""
    
    @staticmethod
    def parse_cameras(filepath: str) -> Dict[int, Camera]:
        cameras = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(x) for x in parts[4:]]
                cameras[camera_id] = Camera(camera_id, model, width, height, params)
        return cameras
    
    @staticmethod
    def parse_frames(filepath: str) -> Dict[int, Frame]:
        frames = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                frame_id = int(parts[0])
                rig_id = int(parts[1])
                quaternion = [float(x) for x in parts[2:6]]
                translation = [float(x) for x in parts[6:9]]
                num_data_ids = int(parts[9])
                data_ids = []
                
                # El formato parece ser: ... num_data_ids SENSOR_TYPE SENSOR_ID DATA_ID
                # En tu ejemplo: 1 CAMERA 1 1
                idx = 10
                for i in range(num_data_ids):
                    if idx + 2 < len(parts):
                        # parts[idx] es el tipo de sensor (ej: "CAMERA")
                        # Necesitamos convertir el string a un número o manejarlo diferente
                        sensor_type_str = parts[idx]
                        sensor_type = 0 if sensor_type_str == "CAMERA" else 1  # Mapeo simple
                        sensor_id = int(parts[idx + 1])
                        data_id = int(parts[idx + 2])
                        data_ids.append((sensor_type, sensor_id, data_id))
                        idx += 3
                    else:
                        break
                
                frames[frame_id] = Frame(frame_id, rig_id, quaternion, translation, data_ids)
        return frames
    
    @staticmethod
    def parse_images(filepath: str) -> Dict[int, Image]:
        images = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or not line:
                i += 1
                continue
            
            # Primera línea: información de la imagen
            parts = line.split()
            image_id = int(parts[0])
            quaternion = [float(x) for x in parts[1:5]]
            translation = [float(x) for x in parts[5:8]]
            camera_id = int(parts[8])
            name = parts[9]
            
            # Segunda línea: puntos 2D
            points2d = []
            if i + 1 < len(lines):
                points_line = lines[i + 1].strip()
                if points_line and not points_line.startswith('#'):
                    points_parts = points_line.split()
                    for j in range(0, len(points_parts), 3):
                        if j + 2 < len(points_parts):
                            x = float(points_parts[j])
                            y = float(points_parts[j + 1])
                            point3d_id = int(points_parts[j + 2])
                            points2d.append((x, y, point3d_id))
            
            images[image_id] = Image(image_id, quaternion, translation, camera_id, name, points2d)
            i += 2
        
        return images
    
    @staticmethod
    def parse_points3d(filepath: str) -> Dict[int, Point3D]:
        points3d = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                point_id = int(parts[0])
                xyz = [float(x) for x in parts[1:4]]
                rgb = [int(x) for x in parts[4:7]]
                error = float(parts[7])
                
                # Parse track
                track = []
                for i in range(8, len(parts), 2):
                    if i + 1 < len(parts):
                        image_id = int(parts[i])
                        point2d_idx = int(parts[i + 1])
                        track.append((image_id, point2d_idx))
                
                points3d[point_id] = Point3D(point_id, xyz, rgb, error, track)
        
        return points3d
    
    @staticmethod
    def parse_rigs(filepath: str) -> Dict[int, Rig]:
        rigs = {}
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                rig_id = int(parts[0])
                num_sensors = int(parts[1])
                
                # Manejar el tipo de sensor como string
                ref_sensor_type_str = parts[2]
                ref_sensor_type = 0 if ref_sensor_type_str == "CAMERA" else 1  # Mapeo simple
                ref_sensor_id = int(parts[3])
                
                sensors = []
                # Parse sensors data (formato complejo, simplificado aquí)
                sensors_data = parts[4:] if len(parts) > 4 else []
                rigs[rig_id] = Rig(rig_id, num_sensors, ref_sensor_type, ref_sensor_id, sensors)
        
        return rigs
    
class SFMQualityAnalyzer:
    """Analizador de calidad para reconstrucciones SFM"""
    
    def __init__(self, normal_path: str, optimized_path: str):
        self.normal_path = normal_path
        self.optimized_path = optimized_path
        self.normal_data = self._load_data(normal_path)
        self.optimized_data = self._load_data(optimized_path)
    
    def _load_data(self, base_path: str) -> Dict:
        """Carga todos los datos de una reconstrucción SFM"""
        sparse_path = os.path.join(base_path, "sparse_txt")
        
        data = {
            'cameras': SFMDataParser.parse_cameras(os.path.join(sparse_path, "cameras.txt")),
            'frames': SFMDataParser.parse_frames(os.path.join(sparse_path, "frames.txt")),
            'images': SFMDataParser.parse_images(os.path.join(sparse_path, "images.txt")),
            'points3d': SFMDataParser.parse_points3d(os.path.join(sparse_path, "points3D.txt")),
            'rigs': SFMDataParser.parse_rigs(os.path.join(sparse_path, "rigs.txt"))
        }
        
        return data
    
    def calculate_reprojection_error(self, data: Dict) -> Dict:
        """Calcula métricas de error de reproyección"""
        points3d = data['points3d']
        
        reprojection_errors = [point.error for point in points3d.values()]
        
        metrics = {
            'mean_error': np.mean(reprojection_errors),
            'median_error': np.median(reprojection_errors),
            'std_error': np.std(reprojection_errors),
            'max_error': np.max(reprojection_errors),
            'min_error': np.min(reprojection_errors),
            'percentile_95': np.percentile(reprojection_errors, 95),
            'percentile_99': np.percentile(reprojection_errors, 99),
            'num_points': len(reprojection_errors),
            'errors': reprojection_errors
        }
        
        return metrics
    
    def calculate_map_completeness(self, data: Dict) -> Dict:
        """Calcula métricas de completitud del mapa 3D"""
        points3d = data['points3d']
        images = data['images']
        
        # Número de puntos 3D
        num_points_3d = len(points3d)
        
        # Número de observaciones totales
        total_observations = sum(len(point.track) for point in points3d.values())
        
        # Longitud media de tracks
        track_lengths = [len(point.track) for point in points3d.values()]
        mean_track_length = np.mean(track_lengths)
        
        # Puntos por imagen
        points_per_image = [len(img.points2d) for img in images.values()]
        mean_points_per_image = np.mean(points_per_image)
        
        # Densidad espacial del mapa
        point_positions = np.array([point.xyz for point in points3d.values()])
        
        # Bounding box del mapa
        bbox_min = np.min(point_positions, axis=0)
        bbox_max = np.max(point_positions, axis=0)
        bbox_size = bbox_max - bbox_min
        map_volume = np.prod(bbox_size)
        
        # Densidad de puntos
        point_density = num_points_3d / map_volume if map_volume > 0 else 0
        
        metrics = {
            'num_points_3d': num_points_3d,
            'total_observations': total_observations,
            'mean_track_length': mean_track_length,
            'std_track_length': np.std(track_lengths),
            'mean_points_per_image': mean_points_per_image,
            'std_points_per_image': np.std(points_per_image),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': bbox_size.tolist(),
            'map_volume': map_volume,
            'point_density': point_density,
            'track_lengths': track_lengths,
            'points_per_image': points_per_image
        }
        
        return metrics
    
    def calculate_precision_metrics(self, data: Dict) -> Dict:
        """Calcula métricas de precisión"""
        points3d = data['points3d']
        images = data['images']
        
        # Análisis de distribución de errores
        errors = [point.error for point in points3d.values()]
        
        # Análisis de consistencia geométrica
        # Calculamos la dispersión de los puntos 3D
        point_positions = np.array([point.xyz for point in points3d.values()])
        
        # Distancia media al centroide
        centroid = np.mean(point_positions, axis=0)
        distances_to_centroid = np.linalg.norm(point_positions - centroid, axis=1)
        
        # Análisis de poses de cámara
        camera_positions = np.array([img.translation for img in images.values()])
        camera_centroid = np.mean(camera_positions, axis=0)
        camera_distances = np.linalg.norm(camera_positions - camera_centroid, axis=1)
        
        metrics = {
            'error_distribution': {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'skewness': float(pd.Series(errors).skew()),
                'kurtosis': float(pd.Series(errors).kurtosis())
            },
            'geometric_consistency': {
                'mean_distance_to_centroid': np.mean(distances_to_centroid),
                'std_distance_to_centroid': np.std(distances_to_centroid),
                'map_centroid': centroid.tolist(),
                'map_spread': np.std(point_positions, axis=0).tolist()
            },
            'camera_geometry': {
                'mean_camera_distance': np.mean(camera_distances),
                'std_camera_distance': np.std(camera_distances),
                'camera_centroid': camera_centroid.tolist(),
                'camera_spread': np.std(camera_positions, axis=0).tolist()
            },
            'num_images': len(images),
            'errors': errors,
            'distances_to_centroid': distances_to_centroid.tolist(),
            'camera_distances': camera_distances.tolist()
        }
        
        return metrics
    
    def compare_reconstructions(self) -> Dict:
        """Compara las dos reconstrucciones y genera métricas de comparación"""
        # Calcular métricas para ambas reconstrucciones
        normal_reprojection = self.calculate_reprojection_error(self.normal_data)
        optimized_reprojection = self.calculate_reprojection_error(self.optimized_data)
        
        normal_completeness = self.calculate_map_completeness(self.normal_data)
        optimized_completeness = self.calculate_map_completeness(self.optimized_data)
        
        normal_precision = self.calculate_precision_metrics(self.normal_data)
        optimized_precision = self.calculate_precision_metrics(self.optimized_data)
        
        # Calcular mejoras
        improvements = self._calculate_improvements(
            normal_reprojection, optimized_reprojection,
            normal_completeness, optimized_completeness,
            normal_precision, optimized_precision
        )
        
        comparison_results = {
            'normal': {
                'reprojection': normal_reprojection,
                'completeness': normal_completeness,
                'precision': normal_precision
            },
            'optimized': {
                'reprojection': optimized_reprojection,
                'completeness': optimized_completeness,
                'precision': optimized_precision
            },
            'improvements': improvements,
            'summary': self._generate_summary(improvements)
        }
        
        return comparison_results
    
    def _calculate_improvements(self, normal_repr, opt_repr, normal_comp, opt_comp, normal_prec, opt_prec) -> Dict:
        """Calcula las mejoras entre las dos reconstrucciones"""
        improvements = {}
        
        # Mejoras en error de reproyección
        improvements['reprojection'] = {
            'mean_error_improvement': ((normal_repr['mean_error'] - opt_repr['mean_error']) / normal_repr['mean_error']) * 100,
            'median_error_improvement': ((normal_repr['median_error'] - opt_repr['median_error']) / normal_repr['median_error']) * 100,
            'std_error_improvement': ((normal_repr['std_error'] - opt_repr['std_error']) / normal_repr['std_error']) * 100,
            'max_error_improvement': ((normal_repr['max_error'] - opt_repr['max_error']) / normal_repr['max_error']) * 100,
        }
        
        # Mejoras en completitud
        improvements['completeness'] = {
            'num_points_change': opt_comp['num_points_3d'] - normal_comp['num_points_3d'],
            'num_points_change_percent': ((opt_comp['num_points_3d'] - normal_comp['num_points_3d']) / normal_comp['num_points_3d']) * 100,
            'track_length_improvement': ((opt_comp['mean_track_length'] - normal_comp['mean_track_length']) / normal_comp['mean_track_length']) * 100,
            'points_per_image_improvement': ((opt_comp['mean_points_per_image'] - normal_comp['mean_points_per_image']) / normal_comp['mean_points_per_image']) * 100,
            'density_improvement': ((opt_comp['point_density'] - normal_comp['point_density']) / normal_comp['point_density']) * 100 if normal_comp['point_density'] > 0 else 0,
        }
        
        # Mejoras en precisión
        improvements['precision'] = {
            'geometric_consistency_improvement': ((normal_prec['geometric_consistency']['std_distance_to_centroid'] - opt_prec['geometric_consistency']['std_distance_to_centroid']) / normal_prec['geometric_consistency']['std_distance_to_centroid']) * 100,
            'camera_stability_improvement': ((normal_prec['camera_geometry']['std_camera_distance'] - opt_prec['camera_geometry']['std_camera_distance']) / normal_prec['camera_geometry']['std_camera_distance']) * 100,
        }
        
        # Test estadístico para diferencias significativas
        if len(normal_repr['errors']) > 0 and len(opt_repr['errors']) > 0:
            t_stat, p_value = ttest_ind(normal_repr['errors'], opt_repr['errors'])
            improvements['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant_improvement': p_value < 0.05 and t_stat > 0
            }
        
        return improvements
    
    def _generate_summary(self, improvements: Dict) -> Dict:
        """Genera un resumen de las mejoras"""
        summary = {
            'overall_quality_improvement': False,
            'key_improvements': [],
            'concerns': []
        }
        
        repr_imp = improvements['reprojection']['mean_error_improvement']
        comp_imp = improvements['completeness']['num_points_change_percent']
        
        if repr_imp > 5:  # Mejora significativa en error de reproyección
            summary['key_improvements'].append(f"Error de reproyección mejorado en {repr_imp:.1f}%")
            summary['overall_quality_improvement'] = True
        
        if comp_imp > 10:  # Aumento significativo en número de puntos
            summary['key_improvements'].append(f"Número de puntos 3D aumentado en {comp_imp:.1f}%")
            summary['overall_quality_improvement'] = True
        
        if comp_imp < -10:  # Disminución significativa en número de puntos
            summary['concerns'].append(f"Número de puntos 3D disminuyó en {abs(comp_imp):.1f}%")
        
        if repr_imp < -5:  # Empeoramiento del error
            summary['concerns'].append(f"Error de reproyección empeoró en {abs(repr_imp):.1f}%")
        
        return summary
    
    def generate_visualizations(self, comparison_results: Dict, output_dir: str):
        """Generate visualizations of the results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Reprojection errors comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error histograms
        normal_errors = comparison_results['normal']['reprojection']['errors']
        opt_errors = comparison_results['optimized']['reprojection']['errors']
        
        ax1.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        ax1.hist(opt_errors, bins=50, alpha=0.7, label='Optimized', density=True)
        ax1.set_xlabel('Reprojection Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Reprojection Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error box plot
        ax2.boxplot([normal_errors, opt_errors], labels=['Normal', 'Optimized'])
        ax2.set_ylabel('Reprojection Error')
        ax2.set_title('Error Comparison (Box Plot)')
        ax2.grid(True, alpha=0.3)
        
        # Completeness metrics
        completeness_metrics = ['num_points_3d', 'mean_track_length', 'mean_points_per_image']
        normal_comp_values = [comparison_results['normal']['completeness'][m] for m in completeness_metrics]
        opt_comp_values = [comparison_results['optimized']['completeness'][m] for m in completeness_metrics]
        
        x = np.arange(len(completeness_metrics))
        width = 0.35
        
        ax3.bar(x - width/2, normal_comp_values, width, label='Normal', alpha=0.8)
        ax3.bar(x + width/2, opt_comp_values, width, label='Optimized', alpha=0.8)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Value')
        ax3.set_title('Completeness Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['3D Points', 'Track Length', 'Points/Image'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Percentage improvements
        improvements_data = comparison_results['improvements']
        improvement_names = ['Mean Error', 'Median Error', '3D Points', 'Track Length']
        improvement_values = [
            improvements_data['reprojection']['mean_error_improvement'],
            improvements_data['reprojection']['median_error_improvement'],
            improvements_data['completeness']['num_points_change_percent'],
            improvements_data['completeness']['track_length_improvement']
        ]
        
        colors = ['green' if x > 0 else 'red' for x in improvement_values]
        ax4.bar(improvement_names, improvement_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Percentage Improvements')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed error analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Q-Q plot to compare distributions
        normal_sorted = np.sort(normal_errors)
        opt_sorted = np.sort(opt_errors)
        
        # Interpolate to have the same number of points
        min_len = min(len(normal_sorted), len(opt_sorted))
        normal_interp = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(normal_sorted)), normal_sorted)
        opt_interp = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(opt_sorted)), opt_sorted)
        
        ax1.scatter(normal_interp, opt_interp, alpha=0.6, s=10)
        ax1.plot([min(normal_interp), max(normal_interp)], [min(normal_interp), max(normal_interp)], 'r--', alpha=0.8)
        ax1.set_xlabel('Normal Errors')
        ax1.set_ylabel('Optimized Errors')
        ax1.set_title('Q-Q Plot: Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Percentile evolution
        percentiles = [50, 75, 90, 95, 99]
        normal_percentiles = [np.percentile(normal_errors, p) for p in percentiles]
        opt_percentiles = [np.percentile(opt_errors, p) for p in percentiles]
        
        ax2.plot(percentiles, normal_percentiles, 'o-', label='Normal', linewidth=2, markersize=8)
        ax2.plot(percentiles, opt_percentiles, 's-', label='Optimized', linewidth=2, markersize=8)
        ax2.set_xlabel('Percentile')
        ax2.set_ylabel('Reprojection Error')
        ax2.set_title('Percentile Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Map completeness analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Track length distribution
        normal_tracks = comparison_results['normal']['completeness']['track_lengths']
        opt_tracks = comparison_results['optimized']['completeness']['track_lengths']
        
        ax1.hist(normal_tracks, bins=30, alpha=0.7, label='Normal', density=True)
        ax1.hist(opt_tracks, bins=30, alpha=0.7, label='Optimized', density=True)
        ax1.set_xlabel('Track Length')
        ax1.set_ylabel('Density')
        ax1.set_title('Track Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Points per image
        normal_ppi = comparison_results['normal']['completeness']['points_per_image']
        opt_ppi = comparison_results['optimized']['completeness']['points_per_image']
        
        ax2.hist(normal_ppi, bins=30, alpha=0.7, label='Normal', density=True)
        ax2.hist(opt_ppi, bins=30, alpha=0.7, label='Optimized', density=True)
        ax2.set_xlabel('Points per Image')
        ax2.set_ylabel('Density')
        ax2.set_title('Points per Image Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bounding box comparison
        normal_bbox = comparison_results['normal']['completeness']['bbox_size']
        opt_bbox = comparison_results['optimized']['completeness']['bbox_size']
        
        bbox_labels = ['X', 'Y', 'Z']
        x = np.arange(len(bbox_labels))
        width = 0.35
        
        ax3.bar(x - width/2, normal_bbox, width, label='Normal', alpha=0.8)
        ax3.bar(x + width/2, opt_bbox, width, label='Optimized', alpha=0.8)
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Size')
        ax3.set_title('Bounding Box Size')
        ax3.set_xticks(x)
        ax3.set_xticklabels(bbox_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary metrics
        summary_data = {
            'Normal': [
                comparison_results['normal']['completeness']['num_points_3d'],
                comparison_results['normal']['completeness']['total_observations'],
                comparison_results['normal']['completeness']['point_density'],
                comparison_results['normal']['completeness']['map_volume']
            ],
            'Optimized': [
                comparison_results['optimized']['completeness']['num_points_3d'],
                comparison_results['optimized']['completeness']['total_observations'],
                comparison_results['optimized']['completeness']['point_density'],
                comparison_results['optimized']['completeness']['map_volume']
            ]
        }
        
        summary_labels = ['3D Points', 'Observations', 'Density', 'Volume']
        
        # Normalize for visualization
        normal_normalized = []
        opt_normalized = []
        for i in range(len(summary_labels)):
            max_val = max(summary_data['Normal'][i], summary_data['Optimized'][i])
            if max_val > 0:
                normal_normalized.append(summary_data['Normal'][i] / max_val)
                opt_normalized.append(summary_data['Optimized'][i] / max_val)
            else:
                normal_normalized.append(0)
                opt_normalized.append(0)
        
        x = np.arange(len(summary_labels))
        ax4.bar(x - width/2, normal_normalized, width, label='Normal', alpha=0.8)
        ax4.bar(x + width/2, opt_normalized, width, label='Optimized', alpha=0.8)
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Completeness Metrics (Normalized)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'completeness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Geometric precision analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distances to centroid
        normal_distances = comparison_results['normal']['precision']['distances_to_centroid']
        opt_distances = comparison_results['optimized']['precision']['distances_to_centroid']
        
        ax1.hist(normal_distances, bins=50, alpha=0.7, label='Normal', density=True)
        ax1.hist(opt_distances, bins=50, alpha=0.7, label='Optimized', density=True)
        ax1.set_xlabel('Distance to Centroid')
        ax1.set_ylabel('Density')
        ax1.set_title('Distance to Centroid Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Camera position scatter plot
        normal_cam_pos = [img.translation for img in self.normal_data['images'].values()]
        opt_cam_pos = [img.translation for img in self.optimized_data['images'].values()]
        
        normal_cam_array = np.array(normal_cam_pos)
        opt_cam_array = np.array(opt_cam_pos)
        
        ax2.scatter(normal_cam_array[:, 0], normal_cam_array[:, 1], alpha=0.6, label='Normal', s=30)
        ax2.scatter(opt_cam_array[:, 0], opt_cam_array[:, 1], alpha=0.6, label='Optimized', s=30)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Camera Positions (XY View)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Precision metrics comparison
        precision_metrics = ['Mean Error', 'Geometric Std', 'Camera Std']
        normal_prec_values = [
            comparison_results['normal']['precision']['error_distribution']['mean'],
            comparison_results['normal']['precision']['geometric_consistency']['std_distance_to_centroid'],
            comparison_results['normal']['precision']['camera_geometry']['std_camera_distance']
        ]
        opt_prec_values = [
            comparison_results['optimized']['precision']['error_distribution']['mean'],
            comparison_results['optimized']['precision']['geometric_consistency']['std_distance_to_centroid'],
            comparison_results['optimized']['precision']['camera_geometry']['std_camera_distance']
        ]
        
        x = np.arange(len(precision_metrics))
        width = 0.35
        
        ax3.bar(x - width/2, normal_prec_values, width, label='Normal', alpha=0.8)
        ax3.bar(x + width/2, opt_prec_values, width, label='Optimized', alpha=0.8)
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Value')
        ax3.set_title('Precision Metrics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(precision_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Improvement heatmap
        improvement_matrix = [
            [improvements_data['reprojection']['mean_error_improvement'],
             improvements_data['reprojection']['median_error_improvement'],
             improvements_data['reprojection']['std_error_improvement']],
            [improvements_data['completeness']['num_points_change_percent'],
             improvements_data['completeness']['track_length_improvement'],
             improvements_data['completeness']['points_per_image_improvement']],
            [improvements_data['precision']['geometric_consistency_improvement'],
             improvements_data['precision']['camera_stability_improvement'],
             0]  # Placeholder to maintain rectangular matrix
        ]
        
        row_labels = ['Reprojection', 'Completeness', 'Precision']
        col_labels = ['Metric 1', 'Metric 2', 'Metric 3']
        
        im = ax4.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
        ax4.set_xticks(range(len(col_labels)))
        ax4.set_yticks(range(len(row_labels)))
        ax4.set_xticklabels(col_labels)
        ax4.set_yticklabels(row_labels)
        ax4.set_title('Improvement Heatmap (%)')
        
        # Add values to heatmap
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax4.text(j, i, f'{improvement_matrix[i][j]:.1f}%',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Improvement (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Executive summary
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Radar chart of main metrics
        categories = ['Mean Error', '3D Points', 'Track Length', 'Geom. Precision', 'Cam. Stability']
        
        # Normalize metrics (invert errors so higher is better)
        normal_radar = [
            1 / (1 + comparison_results['normal']['reprojection']['mean_error']),
            comparison_results['normal']['completeness']['num_points_3d'] / 20000,  # Normalize
            comparison_results['normal']['completeness']['mean_track_length'] / 10,  # Normalize
            1 / (1 + comparison_results['normal']['precision']['geometric_consistency']['std_distance_to_centroid']),
            1 / (1 + comparison_results['normal']['precision']['camera_geometry']['std_camera_distance'])
        ]
        
        opt_radar = [
            1 / (1 + comparison_results['optimized']['reprojection']['mean_error']),
            comparison_results['optimized']['completeness']['num_points_3d'] / 20000,  # Normalize
            comparison_results['optimized']['completeness']['mean_track_length'] / 10,  # Normalize
            1 / (1 + comparison_results['optimized']['precision']['geometric_consistency']['std_distance_to_centroid']),
            1 / (1 + comparison_results['optimized']['precision']['camera_geometry']['std_camera_distance'])
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        normal_radar += normal_radar[:1]
        opt_radar += opt_radar[:1]
        
        ax1 = plt.subplot(2, 2, 1, projection='polar')
        ax1.plot(angles, normal_radar, 'o-', linewidth=2, label='Normal', color='red', alpha=0.7)
        ax1.fill(angles, normal_radar, alpha=0.25, color='red')
        ax1.plot(angles, opt_radar, 's-', linewidth=2, label='Optimized', color='blue', alpha=0.7)
        ax1.fill(angles, opt_radar, alpha=0.25, color='blue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('General Quality Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Main improvements bar chart
        ax2 = plt.subplot(2, 2, 2)
        main_improvements = [
            improvements_data['reprojection']['mean_error_improvement'],
            improvements_data['completeness']['num_points_change_percent'],
            improvements_data['precision']['geometric_consistency_improvement']
        ]
        main_labels = ['Reprojection Error', 'Number of Points', 'Geometric Consistency']
        
        colors = ['green' if x > 0 else 'red' for x in main_improvements]
        bars = ax2.bar(main_labels, main_improvements, color=colors, alpha=0.7)
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Main Improvements')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add values to bars
        for bar, value in zip(bars, main_improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                    f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Summary table
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('off')
        
        summary_data = [
            ['Metric', 'Normal', 'Optimized', 'Improvement'],
            ['Mean Error', f"{comparison_results['normal']['reprojection']['mean_error']:.3f}",
             f"{comparison_results['optimized']['reprojection']['mean_error']:.3f}",
             f"{improvements_data['reprojection']['mean_error_improvement']:.1f}%"],
            ['3D Points', f"{comparison_results['normal']['completeness']['num_points_3d']}",
             f"{comparison_results['optimized']['completeness']['num_points_3d']}",
             f"{improvements_data['completeness']['num_points_change_percent']:.1f}%"],
            ['Track Length', f"{comparison_results['normal']['completeness']['mean_track_length']:.2f}",
             f"{comparison_results['optimized']['completeness']['mean_track_length']:.2f}",
             f"{improvements_data['completeness']['track_length_improvement']:.1f}%"],
            ['Observations', f"{comparison_results['normal']['completeness']['total_observations']}",
             f"{comparison_results['optimized']['completeness']['total_observations']}",
             f"{(comparison_results['optimized']['completeness']['total_observations'] - comparison_results['normal']['completeness']['total_observations']) / comparison_results['normal']['completeness']['total_observations'] * 100:.1f}%"]
        ]
        
        table = ax3.table(cellText=summary_data, cellLoc='center', loc='center', 
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('Summary Table', pad=20, fontsize=14, fontweight='bold')
        
        # Conclusions text
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        summary = comparison_results['summary']
        conclusions_text = "CONCLUSIONS:\n\n"
        
        if summary['overall_quality_improvement']:
            conclusions_text += "✓ OVERALL QUALITY IMPROVEMENT\n\n"
        else:
            conclusions_text += "⚠ MIXED QUALITY RESULTS\n\n"
        
        if summary['key_improvements']:
            conclusions_text += "Key Improvements:\n"
            for imp in summary['key_improvements']:
                conclusions_text += f"• {imp}\n"
            conclusions_text += "\n"
        
        if summary['concerns']:
            conclusions_text += "Considerations:\n"
            for concern in summary['concerns']:
                conclusions_text += f"• {concern}\n"
            conclusions_text += "\n"
        
        # Add statistical significance
        if 'statistical_test' in improvements_data:
            stat_test = improvements_data['statistical_test']
            if stat_test['significant_improvement']:
                conclusions_text += "✓ Statistically significant improvement\n"
            else:
                conclusions_text += "⚠ Improvement not statistically significant\n"
            conclusions_text += f"(p-value: {stat_test['p_value']:.4f})\n"
        
        ax4.text(0.05, 0.95, conclusions_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'executive_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")
    
    def save_results(self, comparison_results: Dict, output_dir: str):
        """Guarda los resultados en formato JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Preparar datos para JSON (convertir arrays numpy a listas)
        json_results = self._prepare_for_json(comparison_results)

         # 2) Purgar recursivamente las claves enormes que no quieres
        keys_to_remove = {
            'errors',
            'track_lengths',
            'points_per_image',
            'distances_to_centroid',
            'camera_distances',
        }

        def purge(d):
            if isinstance(d, dict):
                # eliminar claves primero
                for key in list(d.keys()):
                    if key in keys_to_remove:
                        d.pop(key)
                    else:
                        purge(d[key])
            elif isinstance(d, list):
                for item in d:
                    purge(item)

        purge(json_results)

        
        # Guardar resultados completos
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # Guardar resumen ejecutivo
        summary_report = {
            'dataset': os.path.basename(os.path.dirname(self.normal_path)),
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': comparison_results['summary'],
            'key_metrics': {
                'reprojection_error_improvement': comparison_results['improvements']['reprojection']['mean_error_improvement'],
                'points_3d_change': comparison_results['improvements']['completeness']['num_points_change_percent'],
                'track_length_improvement': comparison_results['improvements']['completeness']['track_length_improvement'],
                'geometric_consistency_improvement': comparison_results['improvements']['precision']['geometric_consistency_improvement']
            },
            'normal_stats': {
                'mean_reprojection_error': comparison_results['normal']['reprojection']['mean_error'],
                'num_points_3d': comparison_results['normal']['completeness']['num_points_3d'],
                'mean_track_length': comparison_results['normal']['completeness']['mean_track_length'],
                'total_observations': comparison_results['normal']['completeness']['total_observations']
            },
            'optimized_stats': {
                'mean_reprojection_error': comparison_results['optimized']['reprojection']['mean_error'],
                'num_points_3d': comparison_results['optimized']['completeness']['num_points_3d'],
                'mean_track_length': comparison_results['optimized']['completeness']['mean_track_length'],
                'total_observations': comparison_results['optimized']['completeness']['total_observations']
            }
        }
        
        with open(os.path.join(output_dir, 'summary_report.json'), 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        # Guardar métricas para análisis posterior
        metrics_csv = pd.DataFrame({
            'metric': ['mean_reprojection_error', 'median_reprojection_error', 'std_reprojection_error',
                      'num_points_3d', 'mean_track_length', 'mean_points_per_image',
                      'total_observations', 'point_density', 'map_volume'],
            'normal': [
                comparison_results['normal']['reprojection']['mean_error'],
                comparison_results['normal']['reprojection']['median_error'],
                comparison_results['normal']['reprojection']['std_error'],
                comparison_results['normal']['completeness']['num_points_3d'],
                comparison_results['normal']['completeness']['mean_track_length'],
                comparison_results['normal']['completeness']['mean_points_per_image'],
                comparison_results['normal']['completeness']['total_observations'],
                comparison_results['normal']['completeness']['point_density'],
                comparison_results['normal']['completeness']['map_volume']
            ],
            'optimized': [
                comparison_results['optimized']['reprojection']['mean_error'],
                comparison_results['optimized']['reprojection']['median_error'],
                comparison_results['optimized']['reprojection']['std_error'],
                comparison_results['optimized']['completeness']['num_points_3d'],
                comparison_results['optimized']['completeness']['mean_track_length'],
                comparison_results['optimized']['completeness']['mean_points_per_image'],
                comparison_results['optimized']['completeness']['total_observations'],
                comparison_results['optimized']['completeness']['point_density'],
                comparison_results['optimized']['completeness']['map_volume']
            ]
        })
        
        metrics_csv.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
        
        print(f"Resultados guardados en: {output_dir}")
    
    def _prepare_for_json(self, data):
        # 1) Convertir arrays completos
        if isinstance(data, np.ndarray):
            return data.tolist()
        
        # 2) Convertir cualquier escalar de NumPy (float, int, bool, …)
        if isinstance(data, np.generic):
            return data.item()
        
        # 3) Recursión en estructuras Python
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._prepare_for_json(v) for v in data]
        
        # 4) Tipos nativos los dejamos igual
        return data
    
    def generate_report(self, output_dir: str):
        """Genera un reporte completo de comparación"""
        print("Iniciando análisis de comparación SFM...")
        
        # Ejecutar análisis
        comparison_results = self.compare_reconstructions()
        
        # Guardar resultados
        self.save_results(comparison_results, output_dir)
        
        # Generar visualizaciones
        self.generate_visualizations(comparison_results, output_dir)
        
        # Generar reporte en texto
        self._generate_text_report(comparison_results, output_dir)
        
        print("Análisis completado exitosamente!")
        return comparison_results
    
    def _generate_text_report(self, comparison_results: Dict, output_dir: str):
        """Genera un reporte en texto plano"""
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE ANÁLISIS DE CALIDAD SFM\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset: {os.path.basename(os.path.dirname(self.normal_path))}\n")
            f.write(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen ejecutivo
            summary = comparison_results['summary']
            f.write("RESUMEN EJECUTIVO\n")
            f.write("-" * 20 + "\n")
            
            if summary['overall_quality_improvement']:
                f.write("✓ MEJORA GENERAL DE CALIDAD DETECTADA\n\n")
            else:
                f.write("⚠ CALIDAD MIXTA - REVISAR DETALLES\n\n")
            
            if summary['key_improvements']:
                f.write("Mejoras Principales:\n")
                for imp in summary['key_improvements']:
                    f.write(f"  • {imp}\n")
                f.write("\n")
            
            if summary['concerns']:
                f.write("Aspectos a Considerar:\n")
                for concern in summary['concerns']:
                    f.write(f"  • {concern}\n")
                f.write("\n")
            
            # Métricas detalladas
            f.write("MÉTRICAS DETALLADAS\n")
            f.write("-" * 20 + "\n\n")
            
            # Error de reproyección
            f.write("1. ERROR DE REPROYECCIÓN\n")
            normal_repr = comparison_results['normal']['reprojection']
            opt_repr = comparison_results['optimized']['reprojection']
            repr_imp = comparison_results['improvements']['reprojection']
            
            f.write(f"   Normal      -> Media: {normal_repr['mean_error']:.4f}, Mediana: {normal_repr['median_error']:.4f}\n")
            f.write(f"   Optimizado  -> Media: {opt_repr['mean_error']:.4f}, Mediana: {opt_repr['median_error']:.4f}\n")
            f.write(f"   Mejora      -> Media: {repr_imp['mean_error_improvement']:.1f}%, Mediana: {repr_imp['median_error_improvement']:.1f}%\n\n")
            
            # Completitud del mapa
            f.write("2. COMPLETITUD DEL MAPA\n")
            normal_comp = comparison_results['normal']['completeness']
            opt_comp = comparison_results['optimized']['completeness']
            comp_imp = comparison_results['improvements']['completeness']
            
            f.write(f"   Normal      -> Puntos 3D: {normal_comp['num_points_3d']}, Track Length: {normal_comp['mean_track_length']:.2f}\n")
            f.write(f"   Optimizado  -> Puntos 3D: {opt_comp['num_points_3d']}, Track Length: {opt_comp['mean_track_length']:.2f}\n")
            f.write(f"   Cambio      -> Puntos 3D: {comp_imp['num_points_change_percent']:.1f}%, Track Length: {comp_imp['track_length_improvement']:.1f}%\n\n")
            
            # Precisión
            f.write("3. PRECISIÓN GEOMÉTRICA\n")
            normal_prec = comparison_results['normal']['precision']
            opt_prec = comparison_results['optimized']['precision']
            prec_imp = comparison_results['improvements']['precision']
            
            f.write(f"   Normal      -> Std Geom: {normal_prec['geometric_consistency']['std_distance_to_centroid']:.4f}\n")
            f.write(f"   Optimizado  -> Std Geom: {opt_prec['geometric_consistency']['std_distance_to_centroid']:.4f}\n")
            f.write(f"   Mejora      -> Consistencia: {prec_imp['geometric_consistency_improvement']:.1f}%\n\n")
            
            # Significancia estadística
            if 'statistical_test' in comparison_results['improvements']:
                stat_test = comparison_results['improvements']['statistical_test']
                f.write("4. SIGNIFICANCIA ESTADÍSTICA\n")
                f.write(f"   t-statistic: {stat_test['t_statistic']:.4f}\n")
                f.write(f"   p-value: {stat_test['p_value']:.6f}\n")
                f.write(f"   Significativo: {'Sí' if stat_test['significant_improvement'] else 'No'}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("Fin del reporte\n")
        
        print(f"Reporte de texto guardado en: {report_path}")

def main():
    """Función principal para ejecutar el análisis"""
    parser = argparse.ArgumentParser(description='Análisis de calidad SFM - Comparación de reconstrucciones')
    parser.add_argument('dataset_number', type=str, help='Número del dataset (ej: 009x, 071x)')
    parser.add_argument('--base_path', type=str, default='.', help='Ruta base donde se encuentran los datasets')
    
    args = parser.parse_args()
    
    # Construir rutas
    dataset_path = os.path.join(args.base_path, args.dataset_number)
    normal_path = os.path.join(dataset_path, 'colmap_normal')
    optimized_path = os.path.join(dataset_path, 'colmap_optimizado')
    results_path = os.path.join(dataset_path, 'results_comparacion')
    
    # Verificar que existan las carpetas
    if not os.path.exists(normal_path):
        print(f"Error: No se encontró la carpeta {normal_path}")
        return
    
    if not os.path.exists(optimized_path):
        print(f"Error: No se encontró la carpeta {optimized_path}")
        return
    
    # Crear analizador y ejecutar análisis
    analyzer = SFMQualityAnalyzer(normal_path, optimized_path)
    results = analyzer.generate_report(results_path)
    
    print(f"\n✓ Análisis completado para dataset {args.dataset_number}")
    print(f"✓ Resultados guardados en: {results_path}")

if __name__ == "__main__":
    main()