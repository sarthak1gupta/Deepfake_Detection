import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, save_dir='checkpoints', device=None):
        self.model = model
        self.save_dir = save_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(images)
            
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
    
    def train(self, train_loader, num_epochs=50, learning_rate=1e-4, weight_decay=1e-5):
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_acc = 0
        for epoch in range(num_epochs):
            avg_loss, accuracy = self.train_epoch(train_loader, optimizer, criterion)
            scheduler.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                self.save_checkpoint(f'best_model.pth')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        print(f"Training completed! Best accuracy: {best_acc:.2f}%")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else None
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(os.path.join(self.save_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('model_config') 