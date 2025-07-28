"""
Deep Learning Models for Tennis Action Recognition
Implements MLP and CNN-based approaches using PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import os


class TennisActionDataset(Dataset):
    """PyTorch Dataset for tennis action recognition"""
    
    def __init__(self, dataframe: pd.DataFrame, transform=None, keypoints_only=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.keypoints_only = keypoints_only
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load keypoints
        keypoints = torch.FloatTensor(row['keypoints'])
        
        # Load image if not keypoints_only
        if not self.keypoints_only:
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
        else:
            image = torch.zeros(3, 224, 224)  # Dummy image
        
        label = torch.LongTensor([row['action_id'] - 1])[0]  # Convert to 0-based
        
        return {
            'image': image,
            'keypoints': keypoints,
            'label': label,
            'image_path': row['image_path']
        }


class KeypointMLP(nn.Module):
    """MLP model for keypoint-based action recognition"""
    
    def __init__(self, input_dim=36, hidden_dims=[512, 256, 128], num_classes=4, dropout=0.3):
        super(KeypointMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, keypoints):
        return self.network(keypoints)


class HybridCNN(nn.Module):
    """Hybrid CNN + MLP model for image + keypoint fusion"""
    
    def __init__(self, num_classes=4, keypoint_dim=36, dropout=0.3):
        super(HybridCNN, self).__init__()
        
        # CNN backbone for images (simplified ResNet-like)
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # MLP for keypoints
        self.keypoint_mlp = nn.Sequential(
            nn.Linear(keypoint_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),  # CNN features + keypoint features
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, keypoints):
        # CNN features
        cnn_features = self.cnn_backbone(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Keypoint features
        keypoint_features = self.keypoint_mlp(keypoints)
        
        # Fusion
        combined_features = torch.cat([cnn_features, keypoint_features], dim=1)
        output = self.fusion(combined_features)
        
        return output


class DeepLearningTrainer:
    """Trainer for deep learning models"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.training_history = {}
        self.action_names = ["backhand", "forehand", "ready_position", "serve"]
        
        print(f"Using device: {self.device}")
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, batch_size: int = 32, 
                          keypoints_only: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        
        # Data transforms
        if not keypoints_only:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = val_test_transform = None
        
        # Create datasets
        train_dataset = TennisActionDataset(train_df, train_transform, keypoints_only)
        val_dataset = TennisActionDataset(val_df, val_test_transform, keypoints_only)
        test_dataset = TennisActionDataset(test_df, val_test_transform, keypoints_only)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   model_name: str, num_epochs: int = 100, learning_rate: float = 0.001,
                   weight_decay: float = 1e-4, patience: int = 10) -> nn.Module:
        """Train a deep learning model"""
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        print(f"Training {model_name}...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch in train_pbar:
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(model, KeypointMLP):
                    outputs = model(keypoints)
                else:  # HybridCNN
                    outputs = model(images, keypoints)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    keypoints = batch['keypoints'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    if isinstance(model, KeypointMLP):
                        outputs = model(keypoints)
                    else:
                        outputs = model(images, keypoints)
                    
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate averages
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        self.models[model_name] = model
        self.training_history[model_name] = history
        
        return model
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, model_name: str) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Evaluating {model_name}'):
                images = batch['image'].to(self.device)
                keypoints = batch['keypoints'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if isinstance(model, KeypointMLP):
                    outputs = model(keypoints)
                else:
                    outputs = model(images, keypoints)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.action_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
        
        print(f'{model_name} Test Results:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  F1 (macro): {f1_macro:.4f}')
        print(f'  F1 (weighted): {f1_weighted:.4f}')
        
        return results
    
    def plot_training_history(self, model_name: str, save_path: str = None):
        """Plot training history"""
        if model_name not in self.training_history:
            print(f"No training history found for {model_name}")
            return
        
        history = self.training_history[model_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'training_history': self.training_history.get(model_name, {})
            }, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save training history
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Example usage of deep learning trainer"""
    # Load processed data
    train_df = pd.read_pickle("data/processed/train_data.pkl")
    val_df = pd.read_pickle("data/processed/val_data.pkl")
    test_df = pd.read_pickle("data/processed/test_data.pkl")
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
    # Train MLP model (keypoints only)
    print("Training MLP model...")
    train_loader_mlp, val_loader_mlp, test_loader_mlp = trainer.create_data_loaders(
        train_df, val_df, test_df, batch_size=32, keypoints_only=True
    )
    
    mlp_model = KeypointMLP(input_dim=36, hidden_dims=[512, 256, 128], num_classes=4)
    trainer.train_model(mlp_model, train_loader_mlp, val_loader_mlp, 'keypoint_mlp', num_epochs=50)
    
    # Train Hybrid CNN model
    print("Training Hybrid CNN model...")
    train_loader_cnn, val_loader_cnn, test_loader_cnn = trainer.create_data_loaders(
        train_df, val_df, test_df, batch_size=16, keypoints_only=False
    )
    
    hybrid_model = HybridCNN(num_classes=4, keypoint_dim=36)
    trainer.train_model(hybrid_model, train_loader_cnn, val_loader_cnn, 'hybrid_cnn', num_epochs=50)
    
    # Evaluate models
    mlp_results = trainer.evaluate_model(mlp_model, test_loader_mlp, 'keypoint_mlp')
    cnn_results = trainer.evaluate_model(hybrid_model, test_loader_cnn, 'hybrid_cnn')
    
    # Plot training history
    trainer.plot_training_history('keypoint_mlp', 'results/mlp_training_history.png')
    trainer.plot_training_history('hybrid_cnn', 'results/cnn_training_history.png')
    
    # Save models
    trainer.save_models("models/deep_learning")


if __name__ == "__main__":
    main()