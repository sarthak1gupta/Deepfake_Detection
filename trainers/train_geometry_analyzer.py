import os
import argparse
from models.geometry_analyzer import create_geometry_analyzer
from models.biological_detector import create_biological_detector
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer
from tqdm import tqdm

class GeometryAnalyzerTrainer(BaseTrainer):
    def __init__(self, model, landmark_extractor, save_dir='checkpoints/geometry_analyzer', device=None):
        super().__init__(model, save_dir, device)
        self.landmark_extractor = landmark_extractor.to(device)
        self.landmark_extractor.eval()  # Set to evaluation mode
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Extract landmarks using the biological detector
            with torch.no_grad():
                landmarks = self.landmark_extractor.extract_facial_landmarks(images)
            
            optimizer.zero_grad()
            output = self.model(landmarks)
            
            if isinstance(output, dict):
                output = output['output']
            
            loss = criterion(output, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(dataloader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Train Geometry 3D Analyzer Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/geometry_analyzer', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--landmark_model_path', type=str, required=True, help='Path to pretrained biological detector for landmark extraction')
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Create landmark extractor (biological detector)
    landmark_extractor = create_biological_detector()
    if os.path.exists(args.landmark_model_path):
        checkpoint = torch.load(args.landmark_model_path, map_location='cpu')
        landmark_extractor.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"Landmark model not found at {args.landmark_model_path}")

    # Create geometry analyzer model
    model = create_geometry_analyzer(
        num_classes=2,
        feature_dim=args.feature_dim
    )

    # Create trainer
    trainer = GeometryAnalyzerTrainer(
        model=model,
        landmark_extractor=landmark_extractor,
        save_dir=args.save_dir
    )

    # Train model
    trainer.train(
        train_loader=train_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

if __name__ == '__main__':
    main() 