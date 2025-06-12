import os
import argparse
import torch
from models.fusion_network import create_fusion_network
from models.texture_analyzer import create_texture_analyzer
from models.frequency_analyzer import create_frequency_analyzer
from models.biological_detector import create_biological_detector
from models.semantic_gnn import create_semantic_gnn
from models.geometry_analyzer import create_geometry_analyzer
from models.attention_maps import create_attention_maps
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer
from tqdm import tqdm

class FusionNetworkTrainer(BaseTrainer):
    def __init__(self, model, component_models, save_dir='checkpoints/fusion_network', device=None):
        super().__init__(model, save_dir, device)
        self.component_models = nn.ModuleDict(component_models)
        for model in self.component_models.values():
            model.to(device)
            model.eval()  # Set to evaluation mode
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Extract features from each component
            component_features = []
            with torch.no_grad():
                # Texture features
                texture_output = self.component_models['texture'](images)
                component_features.append(texture_output['features'])
                
                # Frequency features
                freq_output = self.component_models['frequency'](images)
                component_features.append(freq_output['features'])
                
                # Biological features
                bio_output = self.component_models['biological'](images)
                component_features.append(bio_output['features'])
                
                # Semantic features
                semantic_output = self.component_models['semantic'](bio_output['landmarks'])
                component_features.append(semantic_output['features'])
                
                # Geometry features
                geometry_output = self.component_models['geometry'](bio_output['landmarks'])
                component_features.append(geometry_output['features'])
                
                # Attention features
                attention_output = self.component_models['attention'](images)
                component_features.append(attention_output['features'])
            
            optimizer.zero_grad()
            output = self.model(component_features)
            
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

def load_component_model(model_path, model_creator):
    if os.path.exists(model_path):
        model = model_creator()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Train Dynamic Fusion Network')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/fusion_network', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    
    # Component model paths
    parser.add_argument('--texture_model_path', type=str, required=True, help='Path to pretrained texture analyzer')
    parser.add_argument('--frequency_model_path', type=str, required=True, help='Path to pretrained frequency analyzer')
    parser.add_argument('--biological_model_path', type=str, required=True, help='Path to pretrained biological detector')
    parser.add_argument('--semantic_model_path', type=str, required=True, help='Path to pretrained semantic GNN')
    parser.add_argument('--geometry_model_path', type=str, required=True, help='Path to pretrained geometry analyzer')
    parser.add_argument('--attention_model_path', type=str, required=True, help='Path to pretrained attention maps')
    
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Load component models
    component_models = {
        'texture': load_component_model(args.texture_model_path, create_texture_analyzer),
        'frequency': load_component_model(args.frequency_model_path, create_frequency_analyzer),
        'biological': load_component_model(args.biological_model_path, create_biological_detector),
        'semantic': load_component_model(args.semantic_model_path, create_semantic_gnn),
        'geometry': load_component_model(args.geometry_model_path, create_geometry_analyzer),
        'attention': load_component_model(args.attention_model_path, create_attention_maps)
    }

    # Create fusion network
    input_dims = [512] * len(component_models)  # Each component outputs 512-dim features
    model = create_fusion_network(
        input_dims=input_dims,
        num_classes=2,
        feature_dim=args.feature_dim
    )

    # Create trainer
    trainer = FusionNetworkTrainer(
        model=model,
        component_models=component_models,
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