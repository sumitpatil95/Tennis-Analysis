# Tennis Action Recognition System

A comprehensive machine learning system for recognizing tennis actions from images using both classical ML and deep learning approaches. The system processes COCO-format keypoint annotations and provides a REST API for real-time inference.

## 🎾 Overview

This project implements a complete pipeline for tennis action recognition:

- **Dataset Processing**: COCO format annotation parsing and keypoint extraction
- **Classical ML Models**: Random Forest, SVM, Logistic Regression, K-NN
- **Deep Learning Models**: Keypoint MLP and Hybrid CNN architectures
- **REST API**: FastAPI-based inference server with multiple endpoints
- **Evaluation**: Comprehensive metrics and visualization tools

## 🏗️ Architecture

```
├── src/
│   ├── data/
│   │   └── dataset_processor.py    # COCO annotation processing
│   ├── models/
│   │   ├── classical_ml.py         # Classical ML implementations
│   │   └── deep_learning.py        # PyTorch deep learning models
│   └── api/
│       ├── main.py                 # FastAPI server
│       └── client_example.py       # API client example
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Dataset analysis
│   └── 02_model_training.ipynb     # Model training workflow
├── scripts/
│   ├── train_models.py             # Training script
│   ├── run_api.py                  # API server launcher
│   └── test_api.py                 # API testing script
└── models/                         # Saved model files
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd tennis-action-recognition

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Organize your tennis dataset as follows:
```
data/tennis_dataset/
├── annotations/
│   └── tennis_actions.json        # COCO format annotations
└── images/
    ├── backhand/                  # Backhand shot images
    ├── forehand/                  # Forehand shot images
    ├── ready_position/            # Ready position images
    └── serve/                     # Serve images
```

### 3. Train Models

```bash
# Train all models (classical ML + deep learning)
python scripts/train_models.py --data-dir data/tennis_dataset --evaluate --save-results

# Train only classical ML models
python scripts/train_models.py --model-type classical --classical-models random_forest svm

# Train only deep learning models
python scripts/train_models.py --model-type deep --epochs 100 --batch-size 32
```

### 4. Run API Server

```bash
# Start the API server
python scripts/run_api.py --host 0.0.0.0 --port 8000

# Or use uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test API

```bash
# Test with a sample image
python scripts/test_api.py --image-path data/sample_image.jpg --model-type ensemble

# Test specific model
python scripts/test_api.py --image-path data/sample_image.jpg --model-type classical --model-name random_forest
```

## 📊 Dataset Format

The system expects COCO-format annotations with 18 OpenPose keypoints:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": 18,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {"id": 1, "name": "backhand"},
    {"id": 2, "name": "forehand"},
    {"id": 3, "name": "ready_position"},
    {"id": 4, "name": "serve"}
  ]
}
```

### Keypoint Order (OpenPose format):
1. nose, 2. left_eye, 3. right_eye, 4. left_ear, 5. right_ear
6. left_shoulder, 7. right_shoulder, 8. left_elbow, 9. right_elbow
10. left_wrist, 11. right_wrist, 12. left_hip, 13. right_hip
14. left_knee, 15. right_knee, 16. left_ankle, 17. right_ankle, 18. neck

## 🤖 Models

### Classical ML Models
- **Random Forest**: Ensemble method with hyperparameter tuning
- **SVM**: Support Vector Machine with RBF kernel
- **Logistic Regression**: Linear classifier with regularization
- **K-Nearest Neighbors**: Instance-based learning

### Deep Learning Models
- **Keypoint MLP**: Multi-layer perceptron using only pose features
- **Hybrid CNN**: Combines CNN image features with keypoint MLP

### Feature Engineering
- Normalized keypoint coordinates (36 features)
- Distance-based features (hand distance, shoulder width)
- Angle-based features (arm angles, body orientation)
- Pose center and geometric features

## 🌐 API Endpoints

### Health & Info
- `GET /health` - Health check
- `GET /models/info` - Available models information

### Prediction Endpoints
- `POST /predict/classical` - Classical ML prediction
- `POST /predict/deep` - Deep learning prediction  
- `POST /predict/ensemble` - Ensemble prediction

### Example Usage

```python
import requests

# Predict with ensemble
files = {'file': open('tennis_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict/ensemble', files=files)
result = response.json()

print(f"Predicted action: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1 scores
- **Confusion Matrix**: Per-class performance visualization
- **Classification Report**: Precision, recall, and F1 per class
- **Feature Importance**: For tree-based models

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t tennis-action-recognition .
docker run -p 8000:8000 -v $(pwd)/data:/app/data tennis-action-recognition
```

## 📚 Notebooks

### 1. Data Exploration (`notebooks/01_data_exploration.ipynb`)
- Dataset statistics and visualization
- Keypoint analysis and quality assessment
- Action distribution analysis
- Feature engineering exploration

### 2. Model Training (`notebooks/02_model_training.ipynb`)
- Complete training workflow
- Model comparison and evaluation
- Performance visualization
- Feature importance analysis

## 🔧 Configuration

### Training Parameters
```python
# Classical ML
CLASSICAL_MODELS = ['random_forest', 'svm', 'logistic_regression', 'knn']
HYPERPARAMETER_TUNING = True
CROSS_VALIDATION_FOLDS = 5

# Deep Learning
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
```

### Data Splits
- Training: 60%
- Validation: 20% 
- Test: 20%

## 🚀 Future Extensions

### Sequence-Based Models
- LSTM/GRU for temporal action recognition
- 3D CNN for video-based classification
- Transformer architectures for sequence modeling

### Graph-Based Models
- Graph Convolutional Networks (GCN) for pose graphs
- Spatial-temporal graph modeling
- Attention mechanisms for keypoint relationships

### Advanced Features
- Multi-person action recognition
- Real-time video processing
- Mobile deployment optimization
- Active learning for data annotation

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.9+
- scikit-learn 1.0+
- FastAPI 0.68+
- OpenCV 4.5+
- NumPy, Pandas, Matplotlib

### Optional Dependencies
- Jupyter (for notebooks)
- Docker (for containerization)
- CUDA (for GPU acceleration)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenPose for keypoint detection methodology
- COCO dataset format for annotation standards
- PyTorch and scikit-learn communities
- FastAPI for the excellent web framework

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `notebooks/`
- Review the example scripts in `scripts/`

---

**Happy Tennis Action Recognition! 🎾🤖**