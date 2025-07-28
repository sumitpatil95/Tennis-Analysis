#!/usr/bin/env python3
"""
Training script for tennis action recognition models
Supports both classical ML and deep learning approaches
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_processor import TennisDatasetProcessor
from src.models.classical_ml import ClassicalMLTrainer
from src.models.deep_learning import DeepLearningTrainer, KeypointMLP, HybridCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train tennis action recognition models')
    
    parser.add_argument('--data-dir', type=str, default='data/tennis_dataset',
                       help='Path to tennis dataset directory')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--model-type', type=str, choices=['classical', 'deep', 'all'], 
                       default='all', help='Type of models to train')
    
    # Classical ML arguments
    parser.add_argument('--classical-models', nargs='+', 
                       choices=['random_forest', 'svm', 'logistic_regression', 'knn'],
                       default=['random_forest', 'svm', 'logistic_regression', 'knn'],
                       help='Classical ML models to train')
    
    # Deep learning arguments
    parser.add_argument('--deep-models', nargs='+',
                       choices=['keypoint_mlp', 'hybrid_cnn'],
                       default=['keypoint_mlp', 'hybrid_cnn'],
                       help='Deep learning models to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs for deep learning models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for deep learning models')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for deep learning models')
    
    # Data processing arguments
    parser.add_argument('--annotation-files', nargs='+', 
                       default=['tennis_actions.json'],
                       help='COCO annotation files to process')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (fraction)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set size (fraction)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Evaluation arguments
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate models after training')
    parser.add_argument('--save-results', action='store_true',
                       help='Save evaluation results to JSON')
    
    return parser.parse_args()


def process_dataset(args):
    """Process the tennis dataset"""
    logger.info("Processing tennis dataset...")
    
    processor = TennisDatasetProcessor(args.data_dir)
    
    # Process dataset
    df = processor.process_dataset(args.annotation_files)
    logger.info(f"Processed {len(df)} samples")
    
    # Create splits
    train_df, val_df, test_df = processor.create_train_val_test_split(
        df, test_size=args.test_size, val_size=args.val_size, 
        random_state=args.random_seed
    )
    
    # Save processed data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    processor.save_processed_data(train_df, val_df, test_df, str(processed_dir))
    
    return train_df, val_df, test_df, processor


def train_classical_models(args, train_df, val_df, test_df, processor):
    """Train classical ML models"""
    logger.info("Training classical ML models...")
    
    # Extract features
    X_train, y_train = processor.extract_features_for_classical_ml(train_df)
    X_val, y_val = processor.extract_features_for_classical_ml(val_df)
    X_test, y_test = processor.extract_features_for_classical_ml(test_df)
    
    # Initialize trainer
    trainer = ClassicalMLTrainer()
    
    # Prepare data
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.prepare_data(X_train, X_val, X_test)
    
    # Train models
    results = {}
    
    if 'random_forest' in args.classical_models:
        logger.info("Training Random Forest...")
        trainer.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
        
    if 'svm' in args.classical_models:
        logger.info("Training SVM...")
        trainer.train_svm(X_train_scaled, y_train, X_val_scaled, y_val)
        
    if 'logistic_regression' in args.classical_models:
        logger.info("Training Logistic Regression...")
        trainer.train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)
        
    if 'knn' in args.classical_models:
        logger.info("Training KNN...")
        trainer.train_knn(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate if requested
    if args.evaluate:
        logger.info("Evaluating classical models...")
        results = trainer.evaluate_models(X_test_scaled, y_test)
        
        # Plot results
        trainer.plot_confusion_matrices(results, f"{args.output_dir}/classical_confusion_matrices.png")
        trainer.plot_model_comparison(results, f"{args.output_dir}/classical_comparison.png")
    
    # Save models
    classical_dir = Path(args.output_dir) / "classical_ml"
    classical_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_models(str(classical_dir))
    
    return results


def train_deep_models(args, train_df, val_df, test_df):
    """Train deep learning models"""
    logger.info("Training deep learning models...")
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
    results = {}
    
    if 'keypoint_mlp' in args.deep_models:
        logger.info("Training Keypoint MLP...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_df, val_df, test_df, batch_size=args.batch_size, keypoints_only=True
        )
        
        # Create and train model
        model = KeypointMLP(input_dim=36, hidden_dims=[512, 256, 128], num_classes=4)
        trained_model = trainer.train_model(
            model, train_loader, val_loader, 'keypoint_mlp',
            num_epochs=args.epochs, learning_rate=args.learning_rate
        )
        
        # Evaluate if requested
        if args.evaluate:
            results['keypoint_mlp'] = trainer.evaluate_model(trained_model, test_loader, 'keypoint_mlp')
    
    if 'hybrid_cnn' in args.deep_models:
        logger.info("Training Hybrid CNN...")
        
        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_df, val_df, test_df, batch_size=args.batch_size//2, keypoints_only=False
        )
        
        # Create and train model
        model = HybridCNN(num_classes=4, keypoint_dim=36)
        trained_model = trainer.train_model(
            model, train_loader, val_loader, 'hybrid_cnn',
            num_epochs=args.epochs, learning_rate=args.learning_rate
        )
        
        # Evaluate if requested
        if args.evaluate:
            results['hybrid_cnn'] = trainer.evaluate_model(trained_model, test_loader, 'hybrid_cnn')
    
    # Save models
    deep_dir = Path(args.output_dir) / "deep_learning"
    deep_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_models(str(deep_dir))
    
    return results


def save_evaluation_results(classical_results, deep_results, output_dir):
    """Save evaluation results to JSON"""
    all_results = {
        'classical_ml': classical_results,
        'deep_learning': deep_results
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    # Save results
    results_path = Path(output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")


def main():
    """Main training function"""
    args = parse_arguments()
    
    logger.info("Starting tennis action recognition model training...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    train_df, val_df, test_df, processor = process_dataset(args)
    
    classical_results = {}
    deep_results = {}
    
    # Train models based on type
    if args.model_type in ['classical', 'all']:
        classical_results = train_classical_models(args, train_df, val_df, test_df, processor)
    
    if args.model_type in ['deep', 'all']:
        deep_results = train_deep_models(args, train_df, val_df, test_df)
    
    # Save evaluation results
    if args.save_results and (classical_results or deep_results):
        save_evaluation_results(classical_results, deep_results, args.output_dir)
    
    # Print summary
    logger.info("Training completed successfully!")
    
    if args.evaluate:
        logger.info("\n=== EVALUATION SUMMARY ===")
        
        if classical_results:
            logger.info("\nClassical ML Results:")
            for model_name, results in classical_results.items():
                logger.info(f"  {model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")
        
        if deep_results:
            logger.info("\nDeep Learning Results:")
            for model_name, results in deep_results.items():
                logger.info(f"  {model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_macro']:.4f}")


if __name__ == "__main__":
    main()