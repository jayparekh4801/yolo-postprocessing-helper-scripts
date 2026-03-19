import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
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
        # Distance between first two points and between second and third points
        width1 = np.linalg.norm(points[1] - points[0])
        width2 = np.linalg.norm(points[2] - points[3])
        height1 = np.linalg.norm(points[3] - points[0])
        height2 = np.linalg.norm(points[2] - points[1])
        
        self.width = (width1 + width2) / 2
        self.height = (height1 + height2) / 2
        
    def __repr__(self):
        return f"Detection(class={self.class_id}, conf={self.confidence:.2f}, center={self.center})"


class DronePostProcessor:
    """Post-processing for drone detections using part-based reasoning"""
    
    def __init__(
        self,
        drone_low_conf_threshold: float = 0.5,
        body_high_conf_threshold: float = 0.6,
        body_low_conf_threshold: float = 0.4,
        expansion_factor: float = 1.4,
        search_radius_factor: float = 1.4,
        body_containment_threshold: float = 0.7,
        min_parts_high_conf: int = 1,
        min_parts_low_conf: int = 2
    ):
        """
        Initialize the post-processor with configurable parameters.
        
        Args:
            drone_low_conf_threshold: Drone detections below this are candidates for removal
            body_high_conf_threshold: Body confidence above this requires fewer parts
            body_low_conf_threshold: Body confidence below this requires more parts
            expansion_factor: How much to expand body box to create drone box (e.g., 1.4 = 140%)
            search_radius_factor: Search radius for finding parts near body
            body_containment_threshold: Overlap ratio to consider body "inside" a drone box
            min_parts_high_conf: Min parts needed when body conf is high
            min_parts_low_conf: Min parts needed when body conf is low
        """
        self.drone_low_conf_threshold = drone_low_conf_threshold
        self.body_high_conf_threshold = body_high_conf_threshold
        self.body_low_conf_threshold = body_low_conf_threshold
        self.expansion_factor = expansion_factor
        self.search_radius_factor = search_radius_factor
        self.body_containment_threshold = body_containment_threshold
        self.min_parts_high_conf = min_parts_high_conf
        self.min_parts_low_conf = min_parts_low_conf
        
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
                parts = line.strip().split()
                if len(parts) != 10:  # class_id + 8 coordinates + confidence
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                confidence = float(parts[9])
                
                # Reshape coordinates to 4x2 array (4 points, each with x,y)
                points = np.array(coords).reshape(4, 2)
                
                detections.append(Detection(class_id, points, confidence))
        
        return detections
    
    def save_detections(self, detections: List[Detection], output_path: str):
        """Save detections back to file in OBB format"""
        with open(output_path, 'w') as f:
            for det in detections:
                # Flatten points back to original format
                coords = det.points.flatten()
                line = f"{det.class_id} " + " ".join(f"{x:.6f}" for x in coords) + f" {det.confidence:.6f}\n"
                f.write(line)
    
    def calculate_box_overlap_ratio(self, det1: Detection, det2: Detection) -> float:
        """
        Calculate what percentage of det1's area overlaps with det2.
        Returns: overlap_area / det1_area
        """
        # Get bounding rectangles for simplification
        det1_min = np.min(det1.points, axis=0)
        det1_max = np.max(det1.points, axis=0)
        det2_min = np.min(det2.points, axis=0)
        det2_max = np.max(det2.points, axis=0)
        
        # Calculate intersection
        x_left = max(det1_min[0], det2_min[0])
        y_top = max(det1_min[1], det2_min[1])
        x_right = min(det1_max[0], det2_max[0])
        y_bottom = min(det1_max[1], det2_max[1])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        det1_area = (det1_max[0] - det1_min[0]) * (det1_max[1] - det1_min[1])
        
        if det1_area == 0:
            return 0.0
        
        return intersection_area / det1_area
    
    def is_body_inside_drone(self, body: Detection, drone: Detection) -> bool:
        """Check if body is contained within a drone detection"""
        overlap_ratio = self.calculate_box_overlap_ratio(body, drone)
        return overlap_ratio > self.body_containment_threshold
    
    def find_parts_near_body(
        self, 
        body: Detection, 
        all_parts: List[Detection]
    ) -> List[Detection]:
        """
        Find parts (motors, landing gear) near the body center.
        Uses center-to-center distance with radius based on body dimensions.
        """
        # Calculate search radius
        max_dimension = max(body.width, body.height)
        search_radius = self.search_radius_factor * max_dimension
        
        nearby_parts = []
        for part in all_parts:
            # Calculate center-to-center distance
            distance = np.linalg.norm(part.center - body.center)
            
            if distance < search_radius:
                nearby_parts.append(part)
        
        return nearby_parts
    
    def create_drone_box_from_body(self, body: Detection, parts: List[Detection]) -> Detection:
        """
        Create a new drone detection box from body and parts.
        Expands the body box and assigns average confidence.
        """
        # Calculate average confidence
        all_dets = [body] + parts
        avg_confidence = np.mean([d.confidence for d in all_dets])
        
        # Expand body box by expansion_factor
        center = body.center
        expanded_width = body.width * self.expansion_factor
        expanded_height = body.height * self.expansion_factor
        
        # Create new OBB points (axis-aligned rectangle for simplicity)
        half_w = expanded_width / 2
        half_h = expanded_height / 2
        
        new_points = np.array([
            [center[0] - half_w, center[1] - half_h],  # top-left
            [center[0] + half_w, center[1] - half_h],  # top-right
            [center[0] + half_w, center[1] + half_h],  # bottom-right
            [center[0] - half_w, center[1] + half_h],  # bottom-left
        ])
        
        return Detection(self.DRONE, new_points, avg_confidence)
    
    def has_parts_inside(self, drone: Detection, all_parts: List[Detection]) -> bool:
        """Check if drone box contains any parts (body, motor, landing gear)"""
        drone_min = np.min(drone.points, axis=0)
        drone_max = np.max(drone.points, axis=0)
        
        for part in all_parts:
            # Check if part center is inside drone box
            if (drone_min[0] <= part.center[0] <= drone_max[0] and
                drone_min[1] <= part.center[1] <= drone_max[1]):
                return True
        
        return False
    
    def process(self, input_path: str, output_path: str) -> Dict[str, int]:
        """
        Main processing function.
        
        Returns:
            Dictionary with statistics: removed_drones, added_drones
        """
        # Parse all detections
        all_detections = self.parse_label_file(input_path)
        
        # Separate by class
        drones = [d for d in all_detections if d.class_id == self.DRONE]
        bodies = [d for d in all_detections if d.class_id == self.BODY]
        parts = [d for d in all_detections if d.class_id in [self.MOTOR, self.LANDING_GEAR]]
        
        all_parts = bodies + parts  # For checking parts inside drone
        
        stats = {'removed_drones': 0, 'added_drones': 0}
        
        # ===== PHASE 1: CLEANUP - Remove low-confidence drones with no parts =====
        drones_to_keep = []
        for drone in drones:
            # Check if it's low confidence
            if drone.confidence < self.drone_low_conf_threshold:
                # Check if it has any parts inside
                if not self.has_parts_inside(drone, all_parts):
                    # Remove this drone
                    stats['removed_drones'] += 1
                    continue
            
            drones_to_keep.append(drone)
        
        # ===== PHASE 2: ADD MISSING DRONES - Process orphan bodies =====
        new_drones = []
        
        for body in bodies:
            # Check if body is already inside any drone
            is_inside = False
            for drone in drones_to_keep:
                if self.is_body_inside_drone(body, drone):
                    is_inside = True
                    break
            
            if is_inside:
                continue  # Skip this body, it's already covered
            
            # This is an orphan body - find nearby parts
            nearby_parts = self.find_parts_near_body(body, parts)
            
            # Determine if we should create a drone box
            should_create = False
            
            if body.confidence >= self.body_high_conf_threshold:
                # High confidence body - need min_parts_high_conf parts
                if len(nearby_parts) >= self.min_parts_high_conf:
                    should_create = True
            elif body.confidence >= self.body_low_conf_threshold:
                # Low confidence body - need min_parts_low_conf parts
                if len(nearby_parts) >= self.min_parts_low_conf:
                    should_create = True
            
            if should_create:
                new_drone = self.create_drone_box_from_body(body, nearby_parts)
                new_drones.append(new_drone)
                stats['added_drones'] += 1
        
        # ===== COMBINE AND SAVE =====
        final_detections = drones_to_keep + new_drones + bodies + parts
        self.save_detections(final_detections, output_path)
        
        return stats


def main():
    """Example usage"""
    # Initialize processor with default parameters
    processor = DronePostProcessor(
        drone_low_conf_threshold=0.4,
        body_high_conf_threshold=0.25,
        body_low_conf_threshold=0.4,
        expansion_factor=2,
        search_radius_factor=1.6,
        body_containment_threshold=0.7,
        min_parts_high_conf=1,
        min_parts_low_conf=2
    )

    
    
    # Process a label file
    # input_file = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/postprocessing_testing/AAL3_png.rf.0367394a7f9cbb56cc51092fada3e70b.txt"
    # output_file = "output_labels.txt"

    input_dir = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/xai_results/postprocessed_labels"
    output_dir = "/Users/jaykumarparekh/Documents/Research/drone_postprocessing/yolo_model_testing/xai_results/body_postprocessing_methods_results"

    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file)
            stats = processor.process(input_file, output_file)
            print(f"Processed {file}: Removed drones={stats['removed_drones']}, Added drones={stats['added_drones']}")


if __name__ == "__main__":
    main()