"""
Author : Atharva Jitendra Hude
github: atharvahude
Disclaimer: If it aint broke dont fix it. 
"""

import os
from shapely.geometry import Polygon
import networkx as nx
import json
import logging

# Create a logger
logger = logging.getLogger(__name__)

class XaiPostProcessing:
    def __init__(self, id_name=None, association_dict=None):
        """
        Initialize the XaiPostProcessing class.

        Args:
            id_name (dict): Mapping of class IDs to class names.
            association_dict (dict): Mapping of main classes to their associated subclasses.
        """
        logger.info("Initializing XaiPostProcessing")
        self.id_name = id_name
        self.association_dict = {int(key): value for key, value in association_dict.items()}
        self.main_classes = list(association_dict.keys())
        logger.info(f"Main classes configured: {self.main_classes}")
        
        #Setting up reverse association dict
        self.reverse_association_dict = {}
        for key,value in self.association_dict.items():
            for v in value:
                self.reverse_association_dict[v] = key
        logger.info("Reverse association dictionary created")

    def process_folder(self, input_folder, output_folder):
        """
        Process all files in the input folder and save results to the output folder.
        """
        logger.info(f"Processing folder: {input_folder} -> {output_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            logger.info(f"Created output directory: {output_folder}")

        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                full_path = os.path.join(input_folder, filename)
                logger.info(f"Processing file: {filename}")
                try:
                    with open(full_path, 'r') as f:
                        lines = f.readlines()
                    entries = [self.parse_line(line) for line in lines]
                    final_entries, deleted_count = self.post_processing_logic(entries)
                    logger.info(f"Post-processing removed {deleted_count} entries from {filename}")
                    
                    output_path = os.path.join(output_folder, filename)
                    with open(output_path, 'w') as f:
                        for entry in final_entries:
                            f.write(self.format_line(entry))
                    logger.info(f"Successfully processed and saved: {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)

    def post_processing_logic(self, entries):
        """
        Apply post-processing logic to entries.
    
        Args:
            entries (list): The list of entries to filter.
    
        Returns:
            tuple: (final_entries, number_of_entries_deleted)
        """
        logger.debug("Starting post-processing logic")
        
        # Filter entries with confidence above 0.25 and below 0.25
        entries, low_conf_entries = self.filter_above25_confidence(entries)
        logger.debug(f"Split entries into {len(entries)} high confidence and {len(low_conf_entries)} low confidence")

        # Get the main classes and subclasses
        main_classes = list(self.association_dict.keys())
        sub_classes = []
        for key in self.association_dict:
            sub_classes.extend(self.association_dict[key])
        logger.debug(f"Processing {len(main_classes)} main classes and {len(sub_classes)} subclasses")

        # Get indices of main and sub classes
        main_class_indices = []
        sub_class_indices = []
        for i, entry in enumerate(entries):
            if entry[0] in self.association_dict.keys():
                main_class_indices.append(i)
            if entry[0] in sub_classes:
                sub_class_indices.append(i)
        logger.debug(f"Found {len(main_class_indices)} main class entries and {len(sub_class_indices)} subclass entries")

        # Get entries with low conf main and sub classes
        low_conf_main_class_indices = []
        low_conf_sub_class_indices = []
        for i, entry in enumerate(low_conf_entries):
            if entry[0] in self.association_dict.keys():
                low_conf_main_class_indices.append(i)
            if entry[0] in sub_classes:
                low_conf_sub_class_indices.append(i)
        logger.debug(f"Found {len(low_conf_main_class_indices)} low confidence main entries and {len(low_conf_sub_class_indices)} low confidence subentries")

        # Get entries with part classes entries
        part_class_indices = {key: [] for key in self.association_dict.keys()}
        for i in sub_class_indices:
            for main_class in main_classes:
                if entries[i][0] in self.association_dict[main_class]:
                    part_class_indices[main_class].append(i)
                    break
        logger.debug("Created part class indices mapping")

        # Build graph for overlapping main classes
        G = nx.Graph()
        for i in main_class_indices:
            for j in main_class_indices:
                if i != j and self.calculate_overlap(entries[i][1], entries[j][1], overlap_confidence=0.7):
                    G.add_edge(i, j)
        
        overlapping_sets = list(nx.connected_components(G))
        contestant_list = [list(item) for item in overlapping_sets]
        logger.debug(f"Found {len(contestant_list)} groups of overlapping main classes")

        # Process voting
        voters_registry = {element: [] for sublist in contestant_list for element in sublist}
        winners = []
        
        for contestants in contestant_list:
            confidence_vote = []
            for contestant in contestants:
                id_of_contestant = entries[contestant][0]
                contestant_conf = entries[contestant][2]
                associated_sub_classes = self.association_dict[id_of_contestant]
                
                for i in sub_class_indices:
                    if entries[i][0] in associated_sub_classes:
                        if self.calculate_overlap(entries[contestant][1], entries[i][1], overlap_confidence=0.9):
                            contestant_conf += entries[i][2]
                            voters_registry[contestant].append(i)
                
                confidence_vote.append(contestant_conf)
            
            max_index = confidence_vote.index(max(confidence_vote))
            winners.append(contestants[max_index])
        logger.debug(f"Selected {len(winners)} winners from contestant groups")

        # Process losers
        losers = []
        for contestants in contestant_list:
            for contestant in contestants:
                if contestant not in winners:
                    losers.append(contestant)
                    losers.extend(voters_registry[contestant])

        # Clean indices
        for i in losers:
            if i in main_class_indices:
                main_class_indices.remove(i)
            if i in sub_class_indices:
                sub_class_indices.remove(i)
        logger.debug(f"Removed {len(losers)} losing entries")

        # Process lonely boxes
        lonely_main_boxes = []
        for i in main_class_indices:
            if not any(self.calculate_overlap(entries[i][1], entries[j][1], overlap_confidence=0.7) for j in sub_class_indices):
                lonely_main_boxes.append(i)
        logger.debug(f"Found {len(lonely_main_boxes)} lonely main boxes")

        # Find children for lonely parents
        final_entries = []
        for i in lonely_main_boxes:
            for j in low_conf_sub_class_indices:
                if (self.calculate_overlap(entries[i][1], low_conf_entries[j][1], overlap_confidence=0.7) and 
                    low_conf_entries[j][0] in self.association_dict[entries[i][0]]):
                    final_entries.append(low_conf_entries[j])
                    break

        # Process lonely children
        lonely_children = []
        for i in sub_class_indices:
            if not any(self.calculate_overlap(entries[j][1], entries[i][1], overlap_confidence=0.7) and 
                      entries[j][0] == self.reverse_association_dict[entries[i][0]] for j in main_class_indices):
                lonely_children.append(i)
        logger.debug(f"Found {len(lonely_children)} lonely children")

        # Process low confidence parents
        low_conf_parent_already_taken = {i: False for i in low_conf_main_class_indices}
        low_conf_parent_indices = []

        for i in lonely_children:
            for j in low_conf_main_class_indices:
                if (self.calculate_overlap(low_conf_entries[j][1], entries[i][1], overlap_confidence=0.7) and 
                    low_conf_entries[j][0] == self.reverse_association_dict[entries[i][0]] and 
                    not low_conf_parent_already_taken[j]):
                    low_conf_parent_already_taken[j] = True
                    low_conf_parent_indices.append(j)
                    break

        # Process low confidence parent overlaps
        G = nx.Graph()
        for i in low_conf_parent_indices:
            for j in low_conf_parent_indices:
                if i != j and self.calculate_overlap(low_conf_entries[i][1], low_conf_entries[j][1], overlap_confidence=0.7):
                    G.add_edge(i, j)

        overlapping_sets = list(nx.connected_components(G))
        contestant_list = [list(item) for item in overlapping_sets]
        logger.debug(f"Found {len(contestant_list)} groups of overlapping low confidence parents")

        # Process low confidence voting
        voters_registry = {element: [] for sublist in contestant_list for element in sublist}
        winners = []

        for contestants in contestant_list:
            confidence_vote = []
            for contestant in contestants:
                id_of_contestant = low_conf_entries[contestant][0]
                contestant_conf = low_conf_entries[contestant][2]
                associated_sub_classes = self.association_dict[id_of_contestant]
                
                for i in lonely_children:
                    if entries[i][0] in associated_sub_classes:
                        if self.calculate_overlap(low_conf_entries[contestant][1], entries[i][1], overlap_confidence=0.9):
                            contestant_conf += entries[i][2]
                            voters_registry[contestant].append(i)
                
                confidence_vote.append(contestant_conf)
            
            if confidence_vote:
                max_index = confidence_vote.index(max(confidence_vote))
                winners.append(contestants[max_index])

        # Process final low confidence losers
        low_conf_parent_losers = []
        for i in voters_registry.keys():
            if i not in winners:
                low_conf_parent_losers.append(i)
                losers.extend(voters_registry[i])

        # Build final entries
        for i in low_conf_parent_indices:
            if i not in low_conf_parent_losers:
                final_entries.append(low_conf_entries[i])

        for i, entry in enumerate(entries):
            if i not in losers:
                final_entries.append(entry)

        total_deleted = len(losers) + len(low_conf_parent_losers)
        logger.info(f"Post-processing complete. Removed {total_deleted} entries total")
        return final_entries, total_deleted

    def filter_above25_confidence(self, entries):
        """
        Filter entries with confidence above 0.25.
        """
        above = [entry for entry in entries if entry[2] >= 0.25]
        below = [entry for entry in entries if entry[2] < 0.25]
        logger.debug(f"Filtered {len(above)} entries above threshold, {len(below)} below threshold")
        return above, below

    def calculate_overlap(self, rect1_coords, rect2_coords, overlap_confidence=0.7):
        """
        Calculate overlap between two polygons.
        rect1_coords must be the main class and rect2_coords must be the part.
        """
        try:
            if rect1_coords is None or rect2_coords is None:
                logger.debug("Skipping overlap calculation due to None coordinates")
                return False

            rect1 = Polygon([(rect1_coords[i]*100, rect1_coords[i + 1]*100) for i in range(0, 8, 2)])
            rect2 = Polygon([(rect2_coords[i]*100, rect2_coords[i + 1]*100) for i in range(0, 8, 2)])
            
            intersection = rect1.intersection(rect2).area
            overlap = intersection / rect2.area
            
            logger.debug(f"Overlap calculation: {overlap:.4f} (threshold: {overlap_confidence})")
            return overlap >= overlap_confidence
        except Exception as e:
            logger.error(f"Error calculating overlap: {str(e)}", exc_info=True)
            return False

    def parse_line(self, line):
        """
        Parse a line from the input file.
        """
        try:
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:-1])) if len(parts) > 2 else None
            confidence = float(parts[-1])
            logger.debug(f"Parsed line: class_id={class_id}, confidence={confidence}")
            return class_id, coords, confidence
        except Exception as e:
            logger.error(f"Error parsing line: {line.strip()}, Error: {str(e)}", exc_info=True)
            raise

    def format_line(self, entry):
        """
        Format an entry as a string line.
        """
        try:
            class_id = entry[0]
            coords = ' '.join(map(str, entry[1])) if entry[1] is not None else ''
            confidence = entry[2]
            return f"{class_id} {coords} {confidence}\n"
        except Exception as e:
            logger.error(f"Error formatting entry: {entry}, Error: {str(e)}", exc_info=True)
            raise

def main(config, input_folder, output_folder):
    logger.info("Starting XAI post-processing")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")

    post_processor = XaiPostProcessing(association_dict=config['association_dict'])
    post_processor.process_folder(input_folder, output_folder)
    logger.info("XAI post-processing complete")

if __name__ == '__main__':
    # config_path = 'configs/logic_configs/xai-logic.json'
    # with open(config_path, "r") as f:
    #     config = json.load(f)

    config = {
        "association_dict" : {
            0: [1, 2, 3],
        }
    }

    input_folder = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/labels'
    output_folder = '/Users/jaykumarparekh/Documents/Research/drone_detection_synthetic_org/inference/videos/yolo11l-obb-xai-model-scale_07_10-drone_moving/postprocessed_labels'

    main(config, input_folder, output_folder)
       