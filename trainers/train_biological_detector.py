import os
import argparse
from models.biological_detector import create_biological_detector
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Biological Cues Detector Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/biological_detector', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Create model
    model = create_biological_detector(
        num_classes=2,
        feature_dim=args.feature_dim
    )

    # Create trainer
    trainer = BaseTrainer(
        model=model,
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