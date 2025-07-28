"""
Example client for Tennis Action Recognition API
Demonstrates how to use the API endpoints
"""

import requests
import json
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, Optional


class TennisActionClient:
    """Client for Tennis Action Recognition API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_models_info(self) -> Dict:
        """Get information about available models"""
        response = requests.get(f"{self.base_url}/models/info")
        return response.json()
    
    def predict_classical(self, image_path: str, model_name: str = "random_forest", 
                         keypoints: Optional[np.ndarray] = None) -> Dict:
        """Predict using classical ML model"""
        
        # Prepare files and data
        files = {'file': open(image_path, 'rb')}
        data = {'model_name': model_name}
        
        if keypoints is not None:
            data['keypoints'] = json.dumps(keypoints.tolist())
        
        # Make request
        response = requests.post(f"{self.base_url}/predict/classical", files=files, data=data)
        files['file'].close()
        
        return response.json()
    
    def predict_deep(self, image_path: str, model_name: str = "hybrid_cnn",
                    keypoints: Optional[np.ndarray] = None) -> Dict:
        """Predict using deep learning model"""
        
        files = {'file': open(image_path, 'rb')}
        data = {'model_name': model_name}
        
        if keypoints is not None:
            data['keypoints'] = json.dumps(keypoints.tolist())
        
        response = requests.post(f"{self.base_url}/predict/deep", files=files, data=data)
        files['file'].close()
        
        return response.json()
    
    def predict_ensemble(self, image_path: str, keypoints: Optional[np.ndarray] = None) -> Dict:
        """Predict using ensemble of models"""
        
        files = {'file': open(image_path, 'rb')}
        data = {}
        
        if keypoints is not None:
            data['keypoints'] = json.dumps(keypoints.tolist())
        
        response = requests.post(f"{self.base_url}/predict/ensemble", files=files, data=data)
        files['file'].close()
        
        return response.json()


def demo_usage():
    """Demonstrate API usage"""
    
    # Initialize client
    client = TennisActionClient()
    
    # Check health
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Get models info
    print("\n=== Models Info ===")
    models_info = client.get_models_info()
    print(json.dumps(models_info, indent=2))
    
    # Example image path (replace with actual path)
    image_path = "data/tennis_dataset/images/sample_image.jpg"
    
    if Path(image_path).exists():
        # Example keypoints (18 points * 2 coordinates = 36 values)
        example_keypoints = np.random.rand(36)  # Mock keypoints
        
        print(f"\n=== Predictions for {image_path} ===")
        
        # Classical ML prediction
        print("\n--- Random Forest Prediction ---")
        rf_result = client.predict_classical(image_path, "random_forest", example_keypoints)
        print(f"Predicted: {rf_result['predicted_class']} (confidence: {rf_result['confidence']:.3f})")
        
        # Deep learning prediction
        print("\n--- Hybrid CNN Prediction ---")
        cnn_result = client.predict_deep(image_path, "hybrid_cnn", example_keypoints)
        print(f"Predicted: {cnn_result['predicted_class']} (confidence: {cnn_result['confidence']:.3f})")
        
        # Ensemble prediction
        print("\n--- Ensemble Prediction ---")
        ensemble_result = client.predict_ensemble(image_path, example_keypoints)
        print(f"Predicted: {ensemble_result['predicted_class']} (confidence: {ensemble_result['confidence']:.3f})")
        print(f"Used {ensemble_result['num_models']} models")
        
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path to test predictions")


if __name__ == "__main__":
    demo_usage()