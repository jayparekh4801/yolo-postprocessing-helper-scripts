import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Set
import os


class Detection:
    """Class to represent a single detection with OBB format"""
    def __init__(self, class_id: int, points: np.ndarray, confidence: float):
        self.class_id = class_id
        self.points = points  # 4x2 array of corner points
        self.confidence = confidence
        
        # Calculate center and dimensions
        self.center = np.mean(points, axis=0)
        
        # Calculate width and height from OBB
        width1 = np.linalg.norm(points[1] - points[0])
        width2 = np.linalg.norm(points[2] - points[3])
        height1 = np.linalg.norm(points[3] - points[0])
        height2 = np.linalg.norm(points[2] - points[1])
        
        self.width = (width1 + width2) / 2
        self.height = (height1 + height2) / 2
        
    def __repr__(self):
        return f"Detection(class={self.class_id}, conf={self.confidence:.2f}, center={self.center})"


class HierarchicalDronePostProcessor:
    """Hierarchical post-processing for drone detections using part-based reasoning"""
    
    def __init__(
        self,
        orphan_drone_threshold: float = 0.6,
        part_containment_threshold: float = 0.7,
        # Body parameters
        body_high_conf_threshold: float = 0.6,
        body_low_conf_threshold: float = 0.4,
        body_expansion_factor: float = 1.4,
        body_search_radius_factor: float = 1.4,
        body_min_parts_high_conf: int = 1,
        body_min_parts_low_conf: int = 2,
        # Motor parameters
        motor_search_radius_factor: float = 6.0,
        motor_min_other_parts: int = 1
    ):
        """
        Initialize the hierarchical post-processor with configurable parameters.
        
        Args:
            orphan_drone_threshold: Keep orphan drones above this confidence
            part_containment_threshold: Overlap ratio to consider part "inside" drone
            body_high_conf_threshold: Body confidence above this requires fewer parts
            body_low_conf_threshold: Body confidence below this requires more parts
            body_expansion_factor: How much to expand body box to create drone box
            body_search_radius_factor: Search radius for finding parts near body
            body_min_parts_high_conf: Min OTHER parts needed when body conf is high
            body_min_parts_low_conf: Min OTHER parts needed when body conf is low
            motor_search_radius_factor: Search radius for finding parts near motor
            motor_min_other_parts: Min OTHER parts needed to create drone from motor
        """
        self.orphan_drone_threshold = orphan_drone_threshold
        self.part_containment_threshold = part_containment_threshold
        
        # Body parameters
        self.body_high_conf_threshold = body_high_conf_threshold
        self.body_low_conf_threshold = body_low_conf_threshold
        self.body_expansion_factor = body_expansion_factor
        self.body_search_radius_factor = body_search_radius_factor
        self.body_min_parts_high_conf = body_min_parts_high_conf
        self.body_min_parts_low_conf = body_min_parts_low_conf
        
        # Motor parameters
        self.motor_search_radius_factor = motor_search_radius_factor
        self.motor_min_other_parts = motor_min_other_parts
        
        # Class IDs
        self.DRONE = 0
        self.BODY = 1
        self.LANDING_GEAR = 2
        self.MOTOR = 3
        
    def parse_label_file(self, file_path: str) -> List[Detection]:
        """Parse label file in OBB format"""
        detections = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) != 10:
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                confidence = float(parts[9])
                
                # Reshape coordinates to 4x2 array
                points = np.array(coords).reshape(4, 2)
                
                detections.append(Detection(class_id, points, confidence))
        
        return detections
    
    def save_detections(self, detections: List[Detection], output_path: str):
        """Save detections back to file in OBB format"""
        with open(output_path, 'w') as f:
            for det in detections:
                coords = det.points.flatten()
                line = f"{det.class_id} " + " ".join(f"{x:.6f}" for x in coords) + f" {det.confidence:.6f}\n"
                f.write(line)
    
    def calculate_overlap_ratio(self, part: Detection, drone: Detection) -> float:
        """
        Calculate what percentage of part's area overlaps with drone.
        Returns: overlap_area / part_area
        """
        # Get bounding rectangles
        part_min = np.min(part.points, axis=0)
        part_max = np.max(part.points, axis=0)
        drone_min = np.min(drone.points, axis=0)
        drone_max = np.max(drone.points, axis=0)
        
        # Calculate intersection
        x_left = max(part_min[0], drone_min[0])
        y_top = max(part_min[1], drone_min[1])
        x_right = min(part_max[0], drone_max[0])
        y_bottom = min(part_max[1], drone_max[1])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        part_area = (part_max[0] - part_min[0]) * (part_max[1] - part_min[1])
        
        if part_area == 0:
            return 0.0
        
        return intersection_area / part_area
    
    def find_children_for_drone(self, drone: Detection, parts: List[Detection]) -> Set[int]:
        """
        Find indices of parts that are inside this drone.
        Returns set of indices.
        """
        children_indices = set()
        
        for idx, part in enumerate(parts):
            overlap = self.calculate_overlap_ratio(part, drone)
            if overlap > self.part_containment_threshold:
                children_indices.add(idx)
        
        return children_indices
    
    def find_parts_in_radius(
        self, 
        center_part: Detection, 
        search_radius: float,
        parts_list: List[Detection],
        exclude_indices: Set[int] = None
    ) -> List[Tuple[int, Detection]]:
        """
        Find parts within radius of center_part.
        Returns list of (index, detection) tuples.
        """
        if exclude_indices is None:
            exclude_indices = set()
        
        found_parts = []
        
        for idx, part in enumerate(parts_list):
            if idx in exclude_indices:
                continue
            
            # Calculate center-to-center distance
            distance = np.linalg.norm(part.center - center_part.center)
            
            if distance < search_radius:
                found_parts.append((idx, part))
        
        return found_parts
    
    def create_drone_from_parts(self, parts: List[Detection], expansion_factor: float = None) -> Detection:
        """
        Create a drone detection from a list of parts.
        Can either expand based on first part or create union box.
        """
        if not parts:
            return None
        
        # Calculate average confidence
        avg_confidence = np.mean([p.confidence for p in parts])
        
        if expansion_factor is not None:
            # Expand based on first part (typically body)
            center = parts[0].center
            expanded_width = parts[0].width * expansion_factor
            expanded_height = parts[0].height * expansion_factor
            
            half_w = expanded_width / 2
            half_h = expanded_height / 2
            
            new_points = np.array([
                [center[0] - half_w, center[1] - half_h],
                [center[0] + half_w, center[1] - half_h],
                [center[0] + half_w, center[1] + half_h],
                [center[0] - half_w, center[1] + half_h],
            ])
        else:
            # Create union box of all parts
            all_points = np.vstack([p.points for p in parts])
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            
            new_points = np.array([
                [min_coords[0], min_coords[1]],
                [max_coords[0], min_coords[1]],
                [max_coords[0], max_coords[1]],
                [min_coords[0], max_coords[1]],
            ])
        
        return Detection(self.DRONE, new_points, avg_confidence)
    
    def process_file(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Process a single label file with hierarchical approach.
        
        Returns:
            Dictionary with statistics
        """
        # Parse all detections
        all_detections = self.parse_label_file(input_path)
        
        # Separate by class
        drones = [d for d in all_detections if d.class_id == self.DRONE and d.confidence >= 0.2]
        all_parts = [d for d in all_detections if d.class_id != self.DRONE and d.confidence >= 0.2]
        
        stats = {
            'matched_drones': 0,
            'kept_orphan_drones': 0,
            'removed_orphan_drones': 0,
            'drones_from_bodies': 0,
            'drones_from_motors': 0
        }
        
        # ===== PHASE 1: SEPARATE & MATCH =====
        matched_drones = []
        used_part_indices = set()
        
        for drone in drones:
            children_indices = self.find_children_for_drone(drone, all_parts)
            
            if children_indices:
                # This drone has children - keep it as-is
                matched_drones.append(drone)
                used_part_indices.update(children_indices)
                stats['matched_drones'] += 1
            # Orphan drones will be handled in Phase 2
        
        # Create lists of orphan drones and orphan parts
        orphan_drones = [d for d in drones if d not in matched_drones]
        orphan_parts = [p for idx, p in enumerate(all_parts) if idx not in used_part_indices]
        
        # ===== PHASE 2: FILTER ORPHAN DRONES =====
        kept_orphan_drones = []
        
        for drone in orphan_drones:
            if drone.confidence >= self.orphan_drone_threshold:
                kept_orphan_drones.append(drone)
                stats['kept_orphan_drones'] += 1
            else:
                stats['removed_orphan_drones'] += 1
        
        # ===== PHASE 3: PROCESS ORPHAN PARTS =====
        # Separate orphan parts by class
        orphan_bodies = [p for p in orphan_parts if p.class_id == self.BODY]
        orphan_motors = [p for p in orphan_parts if p.class_id == self.MOTOR]
        orphan_landing_gears = [p for p in orphan_parts if p.class_id == self.LANDING_GEAR]
        
        # Track which parts have been used
        used_orphan_indices = set()
        new_drones = []
        
        # 3.1 BODY PROCESSING
        for body_idx, body in enumerate(orphan_bodies):
            if body_idx in used_orphan_indices:
                continue
            
            # Calculate search radius
            max_dimension = max(body.width, body.height)
            search_radius = self.body_search_radius_factor * max_dimension
            
            # Find other parts in radius (excluding this body)
            # Search in all orphan parts
            all_orphan_parts = orphan_bodies + orphan_motors + orphan_landing_gears
            
            # Create exclude set (parts already used + current body)
            body_offset = 0
            current_body_idx_in_all = body_idx
            exclude_set = used_orphan_indices.copy()
            exclude_set.add(current_body_idx_in_all)
            
            found_parts = self.find_parts_in_radius(
                body, search_radius, all_orphan_parts, exclude_set
            )
            
            # Check if we should create a drone
            should_create = False
            num_found = len(found_parts)
            
            if body.confidence >= self.body_high_conf_threshold:
                if num_found >= self.body_min_parts_high_conf:
                    should_create = True
            elif body.confidence >= self.body_low_conf_threshold:
                if num_found >= self.body_min_parts_low_conf:
                    should_create = True
            
            if should_create:
                # Create drone from body + found parts
                parts_for_drone = [body] + [p for _, p in found_parts]
                new_drone = self.create_drone_from_parts(
                    parts_for_drone, 
                    expansion_factor=self.body_expansion_factor
                )
                new_drones.append(new_drone)
                stats['drones_from_bodies'] += 1
                
                # Mark parts as used
                used_orphan_indices.add(current_body_idx_in_all)
                for idx, _ in found_parts:
                    used_orphan_indices.add(idx)
        
        # 3.2 MOTOR PROCESSING
        motor_offset = len(orphan_bodies)
        
        for motor_local_idx, motor in enumerate(orphan_motors):
            motor_idx_in_all = motor_offset + motor_local_idx
            
            if motor_idx_in_all in used_orphan_indices:
                continue
            
            # Calculate search radius
            max_dimension = max(motor.width, motor.height)
            search_radius = self.motor_search_radius_factor * max_dimension
            
            # Find other parts in radius (excluding this motor)
            all_orphan_parts = orphan_bodies + orphan_motors + orphan_landing_gears
            
            exclude_set = used_orphan_indices.copy()
            exclude_set.add(motor_idx_in_all)
            
            found_parts = self.find_parts_in_radius(
                motor, search_radius, all_orphan_parts, exclude_set
            )
            
            # Check if we have enough OTHER parts
            if len(found_parts) >= self.motor_min_other_parts:
                # Create union box from motor + found parts
                parts_for_drone = [motor] + [p for _, p in found_parts]
                new_drone = self.create_drone_from_parts(
                    parts_for_drone,
                    expansion_factor=None  # Use union box
                )
                new_drones.append(new_drone)
                stats['drones_from_motors'] += 1
                
                # Mark parts as used
                used_orphan_indices.add(motor_idx_in_all)
                for idx, _ in found_parts:
                    used_orphan_indices.add(idx)
        
        # 3.3 Landing gears - ignore remaining ones
        
        # ===== COMBINE AND SAVE =====
        # Reconstruct all parts list (keep all original parts)
        final_detections = (
            matched_drones + 
            kept_orphan_drones + 
            new_drones + 
            all_parts  # Keep all original parts
        )
        
        self.save_detections(final_detections, output_path)
        
        return stats
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, any]:
        """
        Process all .txt files in input directory and save to output directory.
        
        Returns:
            Dictionary with overall statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all .txt files
        txt_files = list(input_path.glob("*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {input_dir}")
            return {}
        
        # Overall statistics
        overall_stats = {
            'total_files': len(txt_files),
            'matched_drones': 0,
            'kept_orphan_drones': 0,
            'removed_orphan_drones': 0,
            'drones_from_bodies': 0,
            'drones_from_motors': 0
        }
        
        # Process each file
        for txt_file in txt_files:
            input_file = str(txt_file)
            output_file = str(output_path / txt_file.name)
            
            file_stats = self.process_file(input_file, output_file)
            
            # Accumulate statistics
            for key in ['matched_drones', 'kept_orphan_drones', 'removed_orphan_drones', 
                       'drones_from_bodies', 'drones_from_motors']:
                overall_stats[key] += file_stats.get(key, 0)
        
        return overall_stats


def main():
    """Example usage"""
    # Initialize processor with parameters
    processor = HierarchicalDronePostProcessor(
        orphan_drone_threshold=0.6,
        part_containment_threshold=0.8,
        # Body parameters
        body_high_conf_threshold=0.6,
        body_low_conf_threshold=0.4,
        body_expansion_factor=1.4,
        body_search_radius_factor=1.4,
        body_min_parts_high_conf=1,
        body_min_parts_low_conf=2,
        # Motor parameters
        motor_search_radius_factor=6.0,
        motor_min_other_parts=1
    )
    
    # Process directory of label files
    input_directory = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/ai_images_postprocessing_testing/old_postprocessing_method_results/postprocessed_labels"
    output_directory = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/ai_images_postprocessing_testing/body_postprocessing_methods_results/labels"
    
    stats = processor.process_directory(input_directory, output_directory)
    
    print("=" * 70)
    print("HIERARCHICAL DRONE POST-PROCESSING RESULTS")
    print("=" * 70)
    print(f"\nTotal files processed: {stats.get('total_files', 0)}")
    print(f"\nMatched drones (with children): {stats.get('matched_drones', 0)}")
    print(f"Kept orphan drones (high conf): {stats.get('kept_orphan_drones', 0)}")
    print(f"Removed orphan drones (low conf): {stats.get('removed_orphan_drones', 0)}")
    print(f"\nNew drones from bodies: {stats.get('drones_from_bodies', 0)}")
    print(f"New drones from motors: {stats.get('drones_from_motors', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()