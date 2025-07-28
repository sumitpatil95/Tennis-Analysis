# Tennis Action Recognition System

A comprehensive machine learning system for recognizing tennis actions from images using both classical ML and deep learning approaches. The system processes COCO-format keypoint annotations and provides a REST API for real-time inference.

## 🎾 Overview

This project implements a complete pipeline for tennis action recognition:

- **Dataset Processing**: COCO format annotation parsing and keypoint extraction
- **Classical ML Models**: Random Forest, SVM, Logistic Regression, K-NN
- **Deep Learning Models**: Keypoint MLP and Hybrid CNN architectures
- **REST API**: FastAPI-based inference server with multiple endpoints
- **Evaluation**: Comprehensive metrics and visualization tools

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

**Happy Tennis Action Recognition! 🎾🤖**
