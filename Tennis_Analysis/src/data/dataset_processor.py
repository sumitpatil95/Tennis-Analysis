"""
Tennis Action Recognition Dataset Processor
Handles COCO format annotations and keypoint data processing
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TennisDatasetProcessor:
    """Process tennis action recognition dataset with COCO annotations"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.annotations_path = self.dataset_path / "annotations"
        self.images_path = self.dataset_path / "images"
        
        # Tennis action categories
        self.action_categories = {
            1: "backhand",
            2: "forehand", 
            3: "ready_position",
            4: "serve"
        }
        
        # OpenPose keypoint names (18 points)
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"
        ]
        
        self.coco_data = None
        self.processed_data = []
        
    def load_coco_annotations(self, annotation_file: str) -> COCO:
        """Load COCO format annotations"""
        annotation_path = self.annotations_path / annotation_file
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        self.coco_data = COCO(str(annotation_path))
        return self.coco_data
    
    def extract_keypoints(self, annotation: Dict) -> np.ndarray:
        """Extract and normalize keypoints from COCO annotation"""
        keypoints = np.array(annotation['keypoints']).reshape(-1, 3)  # x, y, visibility
        
        # Normalize keypoints to [0, 1] range based on image dimensions
        img_info = self.coco_data.loadImgs(annotation['image_id'])[0]
        width, height = img_info['width'], img_info['height']
        
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, 0] /= width   # normalize x
        normalized_keypoints[:, 1] /= height  # normalize y
        
        # Return only x, y coordinates (18 points * 2 = 36 features)
        return normalized_keypoints[:, :2].flatten()
    
    def load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Load and preprocess image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_augmentation_pipeline(self) -> A.Compose:
        """Create data augmentation pipeline"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def process_dataset(self, annotation_files: List[str]) -> pd.DataFrame:
        """Process complete dataset from COCO annotations"""
        all_data = []
        
        for ann_file in annotation_files:
            print(f"Processing {ann_file}...")
            coco = self.load_coco_annotations(ann_file)
            
            # Get all annotations
            ann_ids = coco.getAnnIds()
            annotations = coco.loadAnns(ann_ids)
            
            for ann in annotations:
                try:
                    # Get image info
                    img_info = coco.loadImgs(ann['image_id'])[0]
                    img_path = self.images_path / img_info['file_name']
                    
                    if not img_path.exists():
                        continue
                    
                    # Extract features
                    keypoints = self.extract_keypoints(ann)
                    action_id = ann['category_id']
                    action_name = self.action_categories.get(action_id, "unknown")
                    
                    # Store data
                    data_point = {
                        'image_path': str(img_path),
                        'action_id': action_id,
                        'action_name': action_name,
                        'keypoints': keypoints,
                        'bbox': ann['bbox'],
                        'num_keypoints': ann.get('num_keypoints', 18)
                    }
                    
                    all_data.append(data_point)
                    
                except Exception as e:
                    print(f"Error processing annotation {ann['id']}: {e}")
                    continue
        
        df = pd.DataFrame(all_data)
        print(f"Processed {len(df)} samples")
        return df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                  test_size: float = 0.2, 
                                  val_size: float = 0.2,
                                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/validation/test split"""
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, test_size=test_size, 
            stratify=df['action_name'], 
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, test_size=val_size_adjusted,
            stratify=train_val['action_name'],
            random_state=random_state
        )
        
        print(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        return train, val, test
    
    def extract_features_for_classical_ml(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for classical ML models"""
        # Keypoint features (36 features: 18 points * 2 coordinates)
        keypoint_features = np.vstack(df['keypoints'].values)
        
        # Additional engineered features
        additional_features = []
        
        for _, row in df.iterrows():
            kpts = row['keypoints'].reshape(-1, 2)
            
            # Distance-based features
            # Distance between hands
            left_wrist = kpts[9]   # left_wrist
            right_wrist = kpts[10] # right_wrist
            hand_distance = np.linalg.norm(left_wrist - right_wrist)
            
            # Distance between shoulders
            left_shoulder = kpts[5]  # left_shoulder
            right_shoulder = kpts[6] # right_shoulder
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            
            # Body center (average of shoulders)
            body_center = (left_shoulder + right_shoulder) / 2
            
            # Angles
            # Arm angles (simplified)
            left_arm_vector = left_wrist - left_shoulder
            right_arm_vector = right_wrist - right_shoulder
            
            left_arm_angle = np.arctan2(left_arm_vector[1], left_arm_vector[0])
            right_arm_angle = np.arctan2(right_arm_vector[1], right_arm_vector[0])
            
            # Pose features
            pose_features = [
                hand_distance,
                shoulder_distance,
                left_arm_angle,
                right_arm_angle,
                body_center[0],  # body center x
                body_center[1],  # body center y
            ]
            
            additional_features.append(pose_features)
        
        additional_features = np.array(additional_features)
        
        # Combine all features
        all_features = np.hstack([keypoint_features, additional_features])
        labels = df['action_id'].values - 1  # Convert to 0-based indexing
        
        return all_features, labels
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, output_dir: str):
        """Save processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        train_df.to_pickle(output_path / "train_data.pkl")
        val_df.to_pickle(output_path / "val_data.pkl")
        test_df.to_pickle(output_path / "test_data.pkl")
        
        print(f"Saved processed data to {output_path}")


def main():
    """Example usage of the dataset processor"""
    processor = TennisDatasetProcessor("data/tennis_dataset")
    
    # Process dataset
    annotation_files = ["tennis_actions.json"]  # Your COCO annotation files
    df = processor.process_dataset(annotation_files)
    
    # Create splits
    train_df, val_df, test_df = processor.create_train_val_test_split(df)
    
    # Save processed data
    processor.save_processed_data(train_df, val_df, test_df, "data/processed")
    
    # Extract features for classical ML
    X_train, y_train = processor.extract_features_for_classical_ml(train_df)
    X_val, y_val = processor.extract_features_for_classical_ml(val_df)
    X_test, y_test = processor.extract_features_for_classical_ml(test_df)
    
    print(f"Feature shape: {X_train.shape}")
    print(f"Label distribution: {np.bincount(y_train)}")


if __name__ == "__main__":
    main()