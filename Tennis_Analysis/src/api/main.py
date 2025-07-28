"""
FastAPI server for Tennis Action Recognition
Provides REST API endpoints for model inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import joblib
import numpy as np
import cv2
from PIL import Image
import io
import json
from typing import Optional, Dict, List
import logging
from pathlib import Path
import uvicorn

# Import model classes
from src.models.deep_learning import KeypointMLP, HybridCNN
from src.data.dataset_processor import TennisDatasetProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tennis Action Recognition API",
    description="API for recognizing tennis actions from images and keypoints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
classical_models = {}
deep_models = {}
scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Action names
ACTION_NAMES = ["backhand", "forehand", "ready_position", "serve"]

# Model configurations
MODEL_CONFIGS = {
    'keypoint_mlp': {'input_dim': 36, 'hidden_dims': [512, 256, 128], 'num_classes': 4},
    'hybrid_cnn': {'num_classes': 4, 'keypoint_dim': 36}
}


class ModelManager:
    """Manages loading and inference of different model types"""
    
    def __init__(self):
        self.classical_models = {}
        self.deep_models = {}
        self.scaler = None
        self.processor = TennisDatasetProcessor("data/tennis_dataset")
        
    def load_classical_models(self, model_dir: str):
        """Load classical ML models"""
        model_dir = Path(model_dir)
        
        # Load scaler
        scaler_path = model_dir / "scaler_standard.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
        
        # Load models
        model_files = {
            'random_forest': 'random_forest.pkl',
            'svm': 'svm.pkl',
            'logistic_regression': 'logistic_regression.pkl',
            'knn': 'knn.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = model_dir / filename
            if model_path.exists():
                self.classical_models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name}")
    
    def load_deep_models(self, model_dir: str):
        """Load deep learning models"""
        model_dir = Path(model_dir)
        
        # Load MLP model
        mlp_path = model_dir / "keypoint_mlp.pth"
        if mlp_path.exists():
            checkpoint = torch.load(mlp_path, map_location=device)
            model = KeypointMLP(**MODEL_CONFIGS['keypoint_mlp'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.deep_models['keypoint_mlp'] = model
            logger.info("Loaded keypoint MLP model")
        
        # Load Hybrid CNN model
        cnn_path = model_dir / "hybrid_cnn.pth"
        if cnn_path.exists():
            checkpoint = torch.load(cnn_path, map_location=device)
            model = HybridCNN(**MODEL_CONFIGS['hybrid_cnn'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.deep_models['hybrid_cnn'] = model
            logger.info("Loaded hybrid CNN model")
    
    def extract_keypoints_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract keypoints from image using OpenPose (mock implementation)"""
        # This is a mock implementation
        # In practice, you would use OpenPose or MediaPipe to extract keypoints
        
        # For demo purposes, return random normalized keypoints
        keypoints = np.random.rand(18, 2)  # 18 keypoints, x,y coordinates
        return keypoints.flatten()
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for deep learning models"""
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        
        # Convert to RGB and normalize
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def predict_classical(self, keypoints: np.ndarray, model_name: str) -> Dict:
        """Make prediction using classical ML model"""
        if model_name not in self.classical_models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.classical_models[model_name]
        
        # Scale features
        if self.scaler:
            keypoints_scaled = self.scaler.transform(keypoints.reshape(1, -1))
        else:
            keypoints_scaled = keypoints.reshape(1, -1)
        
        # Predict
        prediction = model.predict(keypoints_scaled)[0]
        probabilities = model.predict_proba(keypoints_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        result = {
            'predicted_class': ACTION_NAMES[prediction],
            'predicted_class_id': int(prediction),
            'confidence': float(probabilities[prediction]) if probabilities is not None else None,
            'probabilities': {
                ACTION_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
            } if probabilities is not None else None,
            'model_type': 'classical',
            'model_name': model_name
        }
        
        return result
    
    def predict_deep(self, image: torch.Tensor, keypoints: torch.Tensor, model_name: str) -> Dict:
        """Make prediction using deep learning model"""
        if model_name not in self.deep_models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.deep_models[model_name]
        
        with torch.no_grad():
            if model_name == 'keypoint_mlp':
                outputs = model(keypoints)
            else:  # hybrid_cnn
                outputs = model(image, keypoints)
            
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(outputs, dim=1)[0]
        
        result = {
            'predicted_class': ACTION_NAMES[prediction.item()],
            'predicted_class_id': int(prediction.item()),
            'confidence': float(probabilities[prediction.item()]),
            'probabilities': {
                ACTION_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'model_type': 'deep_learning',
            'model_name': model_name
        }
        
        return result


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        # Load classical models
        model_manager.load_classical_models("models/classical_ml")
        
        # Load deep learning models
        model_manager.load_deep_models("models/deep_learning")
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tennis Action Recognition API",
        "version": "1.0.0",
        "available_endpoints": [
            "/predict/classical",
            "/predict/deep",
            "/predict/ensemble",
            "/models/info",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "classical": list(model_manager.classical_models.keys()),
            "deep_learning": list(model_manager.deep_models.keys())
        }
    }


@app.get("/models/info")
async def get_models_info():
    """Get information about available models"""
    return {
        "classical_models": {
            name: {
                "type": "classical_ml",
                "algorithm": name.replace('_', ' ').title()
            } for name in model_manager.classical_models.keys()
        },
        "deep_learning_models": {
            name: {
                "type": "deep_learning",
                "architecture": name.replace('_', ' ').title()
            } for name in model_manager.deep_models.keys()
        },
        "action_classes": ACTION_NAMES
    }


@app.post("/predict/classical")
async def predict_classical(
    file: UploadFile = File(...),
    model_name: str = Form("random_forest"),
    keypoints: Optional[str] = Form(None)
):
    """Predict tennis action using classical ML models"""
    try:
        # Validate model
        if model_name not in model_manager.classical_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not available. Available models: {list(model_manager.classical_models.keys())}"
            )
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Extract or use provided keypoints
        if keypoints:
            try:
                keypoints_array = np.array(json.loads(keypoints))
                if keypoints_array.shape != (36,):
                    raise ValueError("Keypoints must be a flat array of 36 values (18 points * 2 coordinates)")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid keypoints format: {e}")
        else:
            # Extract keypoints from image (mock implementation)
            keypoints_array = model_manager.extract_keypoints_from_image(image_np)
        
        # Make prediction
        result = model_manager.predict_classical(keypoints_array, model_name)
        
        # Add metadata
        result.update({
            'image_filename': file.filename,
            'image_size': image.size,
            'keypoints_provided': keypoints is not None
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in classical prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/deep")
async def predict_deep(
    file: UploadFile = File(...),
    model_name: str = Form("hybrid_cnn"),
    keypoints: Optional[str] = Form(None)
):
    """Predict tennis action using deep learning models"""
    try:
        # Validate model
        if model_name not in model_manager.deep_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not available. Available models: {list(model_manager.deep_models.keys())}"
            )
        
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Preprocess image for deep learning
        image_tensor = model_manager.preprocess_image(image_np)
        
        # Extract or use provided keypoints
        if keypoints:
            try:
                keypoints_array = np.array(json.loads(keypoints))
                if keypoints_array.shape != (36,):
                    raise ValueError("Keypoints must be a flat array of 36 values")
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid keypoints format: {e}")
        else:
            keypoints_array = model_manager.extract_keypoints_from_image(image_np)
        
        keypoints_tensor = torch.FloatTensor(keypoints_array).unsqueeze(0)
        
        # Make prediction
        result = model_manager.predict_deep(image_tensor, keypoints_tensor, model_name)
        
        # Add metadata
        result.update({
            'image_filename': file.filename,
            'image_size': image.size,
            'keypoints_provided': keypoints is not None
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in deep learning prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/ensemble")
async def predict_ensemble(
    file: UploadFile = File(...),
    keypoints: Optional[str] = Form(None)
):
    """Predict using ensemble of multiple models"""
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Preprocess for different model types
        image_tensor = model_manager.preprocess_image(image_np)
        
        # Extract or use provided keypoints
        if keypoints:
            try:
                keypoints_array = np.array(json.loads(keypoints))
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid keypoints format: {e}")
        else:
            keypoints_array = model_manager.extract_keypoints_from_image(image_np)
        
        keypoints_tensor = torch.FloatTensor(keypoints_array).unsqueeze(0)
        
        # Get predictions from all available models
        predictions = {}
        all_probabilities = []
        
        # Classical models
        for model_name in model_manager.classical_models.keys():
            try:
                result = model_manager.predict_classical(keypoints_array, model_name)
                predictions[model_name] = result
                if result['probabilities']:
                    all_probabilities.append(list(result['probabilities'].values()))
            except Exception as e:
                logger.warning(f"Error with {model_name}: {e}")
        
        # Deep learning models
        for model_name in model_manager.deep_models.keys():
            try:
                result = model_manager.predict_deep(image_tensor, keypoints_tensor, model_name)
                predictions[model_name] = result
                all_probabilities.append(list(result['probabilities'].values()))
            except Exception as e:
                logger.warning(f"Error with {model_name}: {e}")
        
        # Ensemble prediction (average probabilities)
        if all_probabilities:
            ensemble_probs = np.mean(all_probabilities, axis=0)
            ensemble_prediction = np.argmax(ensemble_probs)
            
            ensemble_result = {
                'predicted_class': ACTION_NAMES[ensemble_prediction],
                'predicted_class_id': int(ensemble_prediction),
                'confidence': float(ensemble_probs[ensemble_prediction]),
                'probabilities': {
                    ACTION_NAMES[i]: float(prob) for i, prob in enumerate(ensemble_probs)
                },
                'model_type': 'ensemble',
                'individual_predictions': predictions,
                'num_models': len(all_probabilities)
            }
        else:
            raise HTTPException(status_code=500, detail="No models available for ensemble prediction")
        
        # Add metadata
        ensemble_result.update({
            'image_filename': file.filename,
            'image_size': image.size,
            'keypoints_provided': keypoints is not None
        })
        
        return ensemble_result
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )