#!/usr/bin/env python3
"""
Script to test the Tennis Action Recognition API
"""

import argparse
import sys
import requests
import json
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.client_example import TennisActionClient


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Tennis Action Recognition API')
    
    parser.add_argument('--base-url', type=str, default='http://localhost:8000',
                       help='Base URL of the API server')
    parser.add_argument('--image-path', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--keypoints-file', type=str, default=None,
                       help='Path to JSON file containing keypoints')
    parser.add_argument('--model-type', type=str, 
                       choices=['classical', 'deep', 'ensemble'], 
                       default='ensemble',
                       help='Type of prediction to test')
    parser.add_argument('--model-name', type=str, default='random_forest',
                       help='Specific model name for classical/deep predictions')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results')
    
    return parser.parse_args()


def load_keypoints(keypoints_file):
    """Load keypoints from JSON file"""
    if keypoints_file and Path(keypoints_file).exists():
        with open(keypoints_file, 'r') as f:
            keypoints_data = json.load(f)
        return np.array(keypoints_data)
    return None


def test_health_and_info(client):
    """Test health check and model info endpoints"""
    print("=== Testing Health Check ===")
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Models loaded: {health['models_loaded']}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return False
    
    print("\n=== Testing Models Info ===")
    try:
        info = client.get_models_info()
        print(f"Classical models: {list(info['classical_models'].keys())}")
        print(f"Deep learning models: {list(info['deep_learning_models'].keys())}")
        print(f"Action classes: {info['action_classes']}")
    except Exception as e:
        print(f"Models info failed: {e}")
        return False
    
    return True


def test_prediction(client, args, keypoints):
    """Test prediction endpoints"""
    image_path = args.image_path
    
    if not Path(image_path).exists():
        print(f"Error: Image file not found: {image_path}")
        return False
    
    print(f"\n=== Testing {args.model_type.title()} Prediction ===")
    print(f"Image: {image_path}")
    print(f"Keypoints provided: {'Yes' if keypoints is not None else 'No'}")
    
    try:
        start_time = time.time()
        
        if args.model_type == 'classical':
            result = client.predict_classical(image_path, args.model_name, keypoints)
        elif args.model_type == 'deep':
            result = client.predict_deep(image_path, args.model_name, keypoints)
        else:  # ensemble
            result = client.predict_ensemble(image_path, keypoints)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"  Predicted Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Model Type: {result['model_type']}")
        print(f"  Inference Time: {inference_time:.2f} ms")
        
        if args.verbose:
            print(f"\nDetailed Results:")
            if 'probabilities' in result:
                print("  Class Probabilities:")
                for action, prob in result['probabilities'].items():
                    print(f"    {action}: {prob:.4f}")
            
            if 'individual_predictions' in result:
                print("  Individual Model Predictions:")
                for model_name, pred in result['individual_predictions'].items():
                    print(f"    {model_name}: {pred['predicted_class']} ({pred['confidence']:.4f})")
            
            print(f"  Metadata:")
            print(f"    Image filename: {result.get('image_filename', 'N/A')}")
            print(f"    Image size: {result.get('image_size', 'N/A')}")
            print(f"    Keypoints provided: {result.get('keypoints_provided', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False


def benchmark_performance(client, args, keypoints, num_requests=10):
    """Benchmark API performance"""
    print(f"\n=== Benchmarking Performance ({num_requests} requests) ===")
    
    times = []
    successes = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            
            if args.model_type == 'classical':
                result = client.predict_classical(args.image_path, args.model_name, keypoints)
            elif args.model_type == 'deep':
                result = client.predict_deep(args.image_path, args.model_name, keypoints)
            else:  # ensemble
                result = client.predict_ensemble(args.image_path, keypoints)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
            successes += 1
            
            print(f"  Request {i+1}/{num_requests}: {times[-1]:.2f} ms")
            
        except Exception as e:
            print(f"  Request {i+1}/{num_requests}: FAILED ({e})")
    
    if times:
        print(f"\nBenchmark Results:")
        print(f"  Success rate: {successes}/{num_requests} ({(successes/num_requests)*100:.1f}%)")
        print(f"  Average time: {np.mean(times):.2f} ms")
        print(f"  Min time: {np.min(times):.2f} ms")
        print(f"  Max time: {np.max(times):.2f} ms")
        print(f"  Std deviation: {np.std(times):.2f} ms")


def main():
    """Main testing function"""
    args = parse_arguments()
    
    print("Tennis Action Recognition API Test")
    print("=" * 50)
    
    # Initialize client
    client = TennisActionClient(args.base_url)
    
    # Test health and info endpoints
    if not test_health_and_info(client):
        print("Basic API tests failed. Exiting.")
        sys.exit(1)
    
    # Load keypoints if provided
    keypoints = load_keypoints(args.keypoints_file)
    if keypoints is not None:
        print(f"Loaded keypoints with shape: {keypoints.shape}")
    
    # Test prediction
    if not test_prediction(client, args, keypoints):
        print("Prediction test failed. Exiting.")
        sys.exit(1)
    
    # Benchmark performance
    benchmark_performance(client, args, keypoints, num_requests=5)
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")


if __name__ == "__main__":
    main()