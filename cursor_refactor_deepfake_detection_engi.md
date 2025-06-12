# Refactor deepfake detection engine components
_Exported on 12/06/2025 at 13:35:44 GMT+5:30 from Cursor (1.0.0)_

---

**User**

# main.py - Multi-Modal Deepfake Detection Engine
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import timm
from scipy import signal
import pywt
from sklearn.preprocessing import StandardScaler
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import face_recognition
import dlib
import mediapipe as mp

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, image_size=224):
        self.transform = transform
        self.image_size = image_size
        self.images = []
        self.labels = []
        
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)
        
        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class HighResTextureAnalyzer(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.cnn_backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        vit_dim = self.vit_backbone.num_features
        cnn_dim = self.cnn_backbone.num_features
        
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + cnn_dim + 128 + 64, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        vit_features = self.vit_backbone(x)
        cnn_features = self.cnn_backbone(x)
        texture_features = self.texture_conv(x).flatten(1)
        edge_features = self.edge_detector(x).flatten(1)
        
        combined = torch.cat([vit_features, cnn_features, texture_features, edge_features], dim=1)
        return self.fusion(combined)

class FrequencyDomainInspector(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.spectral_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.wavelet_processor = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 + 512, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def extract_frequency_features(self, x):
        batch_size = x.shape[0]
        freq_features = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            f_transform = np.fft.fft2(img_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            coeffs = pywt.dwt2(img_gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
            
            if len(wavelet_features) > 256 * 4:
                wavelet_features = wavelet_features[:256 * 4]
            elif len(wavelet_features) < 256 * 4:
                wavelet_features = np.pad(wavelet_features, (0, 256 * 4 - len(wavelet_features)))
            
            freq_features.append(wavelet_features)
        
        return torch.FloatTensor(np.array(freq_features)).to(x.device)
    
    def forward(self, x):
        spectral_features = self.spectral_cnn(x).flatten(1)
        wavelet_features = self.extract_frequency_features(x)
        wavelet_processed = self.wavelet_processor(wavelet_features)
        
        combined = torch.cat([spectral_features, wavelet_processed], dim=1)
        return self.fusion(combined)

class BiologicalCuesDetector(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.rppg_analyzer = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.micro_expression_net = nn.Sequential(
            nn.Linear(468 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def extract_facial_landmarks(self, x):
        batch_size = x.shape[0]
        landmark_features = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            img_rgb = (img * 255).astype(np.uint8)
            
            results = self.mp_face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                # Ensure length is always 1404
                if len(coords) < 468 * 3:
                    coords += [0.0] * (468 * 3 - len(coords))
                elif len(coords) > 468 * 3:
                    coords = coords[:468 * 3]
                landmark_features.append(coords)
            else:
                landmark_features.append([0.0] * (468 * 3))
        
        return torch.FloatTensor(np.array(landmark_features)).to(x.device)
    
    def extract_rppg_signal(self, x):
        batch_size = x.shape[0]
        rppg_signals = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            
            face_locations = face_recognition.face_locations((img * 255).astype(np.uint8))
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_roi = img[top:bottom, left:right]
                
                if face_roi.size > 0:
                    rgb_signal = np.mean(face_roi, axis=(0, 1))
                    rppg_signals.append(rgb_signal)
                else:
                    rppg_signals.append([0.0, 0.0, 0.0])
            else:
                rppg_signals.append([0.0, 0.0, 0.0])
        
        rppg_tensor = torch.FloatTensor(np.array(rppg_signals)).to(x.device)
        return rppg_tensor.unsqueeze(2)  # [batch, 3, 1]
    
    def forward(self, x):
        landmark_features = self.extract_facial_landmarks(x)
        rppg_signal = self.extract_rppg_signal(x)
        
        micro_expr_features = self.micro_expression_net(landmark_features)
        rppg_features = self.rppg_analyzer(rppg_signal).flatten(1)
        
        combined = torch.cat([micro_expr_features, rppg_features], dim=1)
        return self.fusion(combined)

class SemanticConsistencyGNN(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.node_encoder = nn.Linear(3, 64)
        self.gnn_layers = nn.ModuleList([
            gnn.GCNConv(64, 128),
            gnn.GCNConv(128, 256),
            gnn.GCNConv(256, 128)
        ])
        self.global_pool = gnn.global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def create_face_graph(self, landmarks):
        batch_graphs = []
        
        for batch_idx in range(landmarks.shape[0]):
            landmark_coords = landmarks[batch_idx].view(-1, 3)
            
            edges = []
            for i in range(len(landmark_coords)):
                for j in range(i+1, min(i+10, len(landmark_coords))):
                    edges.append([i, j])
                    edges.append([j, i])
            
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=landmark_coords, edge_index=edge_index.to(landmarks.device))
            batch_graphs.append(graph)
        
        return batch_graphs
    
    def forward(self, landmarks):
        batch_graphs = self.create_face_graph(landmarks)
        batch_features = []
        
        for graph in batch_graphs:
            x = self.node_encoder(graph.x)
            
            for gnn_layer in self.gnn_layers:
                x = F.relu(gnn_layer(x, graph.edge_index))
            
            graph_feature = self.global_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            batch_features.append(graph_feature)
        
        batch_tensor = torch.stack(batch_features)
        return self.classifier(batch_tensor)

class Geometry3DAnalyzer(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        self.pose_estimator = nn.Sequential(
            nn.Linear(468 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.geometry_net = nn.Sequential(
            nn.Linear(64 + 6, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def estimate_head_pose(self, landmarks):
        batch_size = landmarks.shape[0]
        pose_features = []
        
        for i in range(batch_size):
            landmark_3d = landmarks[i].view(-1, 3).cpu().numpy()
            
            if len(landmark_3d) >= 6:
                nose_tip = landmark_3d[1]
                left_eye = landmark_3d[33]
                right_eye = landmark_3d[263]
                left_mouth = landmark_3d[61]
                right_mouth = landmark_3d[291]
                chin = landmark_3d[18]
                
                pose_vector = np.array([
                    nose_tip[0] - (left_eye[0] + right_eye[0]) / 2,
                    nose_tip[1] - (left_eye[1] + right_eye[1]) / 2,
                    nose_tip[2] - (left_eye[2] + right_eye[2]) / 2,
                    (right_eye[0] - left_eye[0]),
                    (right_mouth[0] - left_mouth[0]),
                    chin[1] - nose_tip[1]
                ])
            else:
                pose_vector = np.zeros(6)
            
            pose_features.append(pose_vector)
        
        return torch.FloatTensor(np.array(pose_features)).to(landmarks.device)
    
    def forward(self, landmarks):
        pose_features = self.estimate_head_pose(landmarks)
        landmark_features = self.pose_estimator(landmarks.view(landmarks.shape[0], -1))
        
        combined = torch.cat([landmark_features, pose_features], dim=1)
        return self.geometry_net(combined)

class AttentionMaps(nn.Module):
    def __init__(self, input_dim, num_classes=512):
        super().__init__()
        attn_channels = max(1, input_dim // 4)
        self.attention_conv = nn.Sequential(
            nn.Conv2d(input_dim, attn_channels, kernel_size=1),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_dim, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        attention_weights = self.attention_conv(x)
        attended_features = x * attention_weights
        features = self.feature_extractor(attended_features)
        return features.flatten(1), attention_weights

class DynamicFusionNetwork(nn.Module):
    def __init__(self, input_dims, num_classes=2):
        super().__init__()
        total_dim = sum(input_dims)
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(input_dims)),
            nn.Softmax(dim=1)
        )
        
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features_list):
        # Ensure all features are [batch, C]
        features_list = [f.flatten(1) if f.dim() > 2 else f for f in features_list]
        combined_features = torch.cat(features_list, dim=1)
        
        uncertainty_weights = self.uncertainty_estimator(combined_features)
        
        weighted_features = []
        start_idx = 0
        for i, features in enumerate(features_list):
            end_idx = start_idx + features.shape[1]
            weight = uncertainty_weights[:, i:i+1]
            weighted_features.append(features * weight)
            start_idx = end_idx
        
        final_features = torch.cat(weighted_features, dim=1)
        
        output = self.fusion_net(final_features)
        confidence = self.confidence_estimator(combined_features)
        
        return output, confidence

class MultiModalDetectionEngine(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.texture_analyzer = HighResTextureAnalyzer(512)
        self.frequency_inspector = FrequencyDomainInspector(512)
        self.biological_detector = BiologicalCuesDetector(512)
        self.semantic_gnn = SemanticConsistencyGNN(512)
        self.geometry_analyzer = Geometry3DAnalyzer(512)
        self.attention_maps = AttentionMaps(3, 512)
        
        self.fusion_network = DynamicFusionNetwork(
            [512, 512, 512, 512, 512, 512], 
            num_classes
        )
    
    def forward(self, x):
        texture_features = self.texture_analyzer(x)
        frequency_features = self.frequency_inspector(x)
        biological_features = self.biological_detector(x)
        
        landmarks = self.biological_detector.extract_facial_landmarks(x)
        semantic_features = self.semantic_gnn(landmarks)
        geometry_features = self.geometry_analyzer(landmarks)
        
        attention_features, attention_weights = self.attention_maps(x)
        
        all_features = [
            texture_features,
            frequency_features, 
            biological_features,
            semantic_features,
            geometry_features,
            attention_features
        ]
        
        final_output, confidence = self.fusion_network(all_features)
        
        return {
            'output': final_output,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'individual_features': {
                'texture': texture_features,
                'frequency': frequency_features,
                'biological': biological_features,
                'semantic': semantic_features,
                'geometry': geometry_features,
                'attention': attention_features
            }
        }

def create_data_loaders(train_real_dir, train_fake_dir, batch_size=16, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DeepfakeDataset(train_real_dir, train_fake_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

def train_model(model, dataloader, num_epochs=50, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            results = model(images)
            output = results['output']
            confidence = results['confidence']
            
            classification_loss = criterion(output, labels)
            confidence_loss = torch.mean((confidence.squeeze() - torch.max(F.softmax(output, dim=1), dim=1)[0]) ** 2)
            
            total_loss_batch = classification_loss + 0.1 * confidence_loss
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {total_loss_batch.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'deepfake_model_epoch_{epoch+1}.pth')

def main():
    train_real_dir = 'Dataset/Train/Real'
    train_fake_dir = 'Dataset/Train/Fake'
    
    dataloader = create_data_loaders(train_real_dir, train_fake_dir, batch_size=8)
    
    model = MultiModalDetectionEngine(num_classes=2)
    
    print("Starting training...")
    train_model(model, dataloader, num_epochs=50, learning_rate=1e-4)
    
    torch.save(model.state_dict(), 'final_deepfake_detection_model.pth')
    print("Training completed and model saved!")

if __name__ == "__main__":
    main()

Refactor this code into componentes because it is not able to train on this as a whole, so I am planning to train on multple components grouped in various components. Datset directory should remain same for both real and fake.

---

**Cursor**

I'll help you refactor this code into separate components that can be trained independently. I'll break it down into logical modules that can be trained separately and then combined. Let me explain the refactoring strategy:

1. First, I'll create a modular structure with separate files for each component
2. Each component will be independently trainable
3. We'll maintain a common dataset structure
4. We'll create a unified interface for combining the models later

Let me start by creating the necessary directory structure and files:


I'll create a modular structure with the following components:

1. `dataset/` - For dataset handling
2. `models/` - For individual model components
3. `trainers/` - For training scripts
4. `utils/` - For utility functions
5. `config/` - For configuration files

Let me create these directories and the necessary files:


```python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, image_size=224):
        self.transform = transform
        self.image_size = image_size
        self.images = []
        self.labels = []
        
        for img_name in os.listdir(real_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(real_dir, img_name))
                self.labels.append(0)
        
        for img_name in os.listdir(fake_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_default_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_data_loader(real_dir, fake_dir, batch_size=16, image_size=224, num_workers=4):
    transform = get_default_transform(image_size)
    dataset = DeepfakeDataset(real_dir, fake_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
```

Now, let's create the base trainer module that will be used by all components:


```python
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
```

Now, let's create the first component model - the texture analyzer:


```python
import torch
import torch.nn as nn
import timm

class HighResTextureAnalyzer(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.vit_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.cnn_backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.edge_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        vit_dim = self.vit_backbone.num_features
        cnn_dim = self.cnn_backbone.num_features
        
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + cnn_dim + 128 + 64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def forward(self, x):
        vit_features = self.vit_backbone(x)
        cnn_features = self.cnn_backbone(x)
        texture_features = self.texture_conv(x).flatten(1)
        edge_features = self.edge_detector(x).flatten(1)
        
        combined = torch.cat([vit_features, cnn_features, texture_features, edge_features], dim=1)
        features = self.fusion(combined)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features
        }

def create_texture_analyzer(num_classes=2, feature_dim=512):
    model = HighResTextureAnalyzer(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let's create a training script for the texture analyzer component:


```python
import os
import argparse
from models.texture_analyzer import create_texture_analyzer
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Texture Analyzer Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/texture_analyzer', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Create model
    model = create_texture_analyzer(
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
```

Let's create a requirements.txt file to manage dependencies:


```pip-requirements
torch>=2.0.0
torchvision>=0.15.0
timm>=0.6.0
numpy>=1.21.0
Pillow>=8.0.0
opencv-python>=4.5.0
scipy>=1.7.0
PyWavelets>=1.2.0
scikit-learn>=0.24.0
face-recognition>=1.3.0
dlib>=19.22.0
mediapipe>=0.8.0
tqdm>=4.62.0
torch-geometric>=2.0.0
```

Finally, let's create a README.md to explain how to use the refactored code:


```markdown
# Multi-Modal Deepfake Detection Engine

This repository contains a modular implementation of a multi-modal deepfake detection system. The system is broken down into independent components that can be trained separately and then combined for final inference.

## Project Structure

```
.
├── dataset/
│   └── dataset.py           # Dataset handling code
├── models/
│   ├── texture_analyzer.py  # Texture analysis component
│   └── ...                  # Other model components
├── trainers/
│   ├── base_trainer.py      # Base training functionality
│   ├── train_texture_analyzer.py  # Texture analyzer training script
│   └── ...                  # Other training scripts
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Individual Components

Each component can be trained independently. Here's how to train the texture analyzer component:

```bash
python trainers/train_texture_analyzer.py \
    --train_real_dir Dataset/Train/Real \
    --train_fake_dir Dataset/Train/Fake \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --save_dir checkpoints/texture_analyzer
```

## Dataset Structure

The dataset should be organized as follows:
```
Dataset/
├── Train/
│   ├── Real/
│   │   ├── real_image1.jpg
│   │   ├── real_image2.jpg
│   │   └── ...
│   └── Fake/
│       ├── fake_image1.jpg
│       ├── fake_image2.jpg
│       └── ...
```

## Component Details

### Texture Analyzer
- Uses ViT and EfficientNet backbones
- Analyzes high-resolution texture patterns
- Detects edge inconsistencies
- Outputs both classification and feature embeddings

## Training Process

1. Each component is trained independently using its own training script
2. Components save their checkpoints in separate directories
3. After training, components can be loaded and combined for inference

## Future Components

The following components will be added:
- Frequency Domain Inspector
- Biological Cues Detector
- Semantic Consistency GNN
- Geometry 3D Analyzer
- Attention Maps
- Dynamic Fusion Network

## Notes

- Each component is designed to be trained independently
- Components can be trained in parallel
- The final model will combine all components using a fusion network
- Checkpoints are saved regularly during training
- Best models are saved separately for each component
```

I've refactored the code into a modular structure where each component can be trained independently. Here's what I've done:

1. Created a modular directory structure:
   - `dataset/` - Contains dataset handling code
   - `models/` - Contains individual model components
   - `trainers/` - Contains training scripts
   - Added `requirements.txt` and `README.md`

2. Implemented the first component (Texture Analyzer) as an example:
   - Created `models/texture_analyzer.py` with the independent model
   - Created `trainers/train_texture_analyzer.py` for training this component
   - Added a base trainer class that can be used by all components

3. Made the code more maintainable:
   - Each component is self-contained
   - Common functionality is shared through base classes
   - Training scripts are separate and configurable
   - Added proper documentation

To train the texture analyzer component, you can now run:
```bash
python trainers/train_texture_analyzer.py \
    --train_real_dir Dataset/Train/Real \
    --train_fake_dir Dataset/Train/Fake
```

Would you like me to implement any of the other components next? I can create:
1. Frequency Domain Inspector
2. Biological Cues Detector
3. Semantic Consistency GNN
4. Geometry 3D Analyzer
5. Attention Maps
6. Dynamic Fusion Network

Each component will follow the same modular pattern and can be trained independently. Let me know which component you'd like me to implement next!

```shellscript
python trainers/train_texture_analyzer.py \
    --train_real_dir Dataset/Train/Real \
    --train_fake_dir Dataset/Train/Fake
```

---

**User**

Implemented the first component (Texture Analyzer) as an example:
Created models/texture_analyzer.py with the independent model
Created trainers/train_texture_analyzer.py for training this component
Added a base trainer class that can be used by all components
create all models codes and in trainers also create the corresponding trainign code

---

**Cursor**

I'll help you create all the remaining model components and their corresponding training scripts. Let's implement them one by one.

First, let's create the Frequency Domain Inspector:


```python
import torch
import torch.nn as nn
import cv2
import numpy as np
import pywt
from scipy import signal

class FrequencyDomainInspector(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.spectral_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.wavelet_processor = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256 + 512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def extract_frequency_features(self, x):
        batch_size = x.shape[0]
        freq_features = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            img_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # FFT analysis
            f_transform = np.fft.fft2(img_gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Wavelet analysis
            coeffs = pywt.dwt2(img_gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            wavelet_features = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
            
            # Ensure consistent feature size
            if len(wavelet_features) > 256 * 4:
                wavelet_features = wavelet_features[:256 * 4]
            elif len(wavelet_features) < 256 * 4:
                wavelet_features = np.pad(wavelet_features, (0, 256 * 4 - len(wavelet_features)))
            
            freq_features.append(wavelet_features)
        
        return torch.FloatTensor(np.array(freq_features)).to(x.device)
    
    def forward(self, x):
        spectral_features = self.spectral_cnn(x).flatten(1)
        wavelet_features = self.extract_frequency_features(x)
        wavelet_processed = self.wavelet_processor(wavelet_features)
        
        combined = torch.cat([spectral_features, wavelet_processed], dim=1)
        features = self.fusion(combined)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features
        }

def create_frequency_analyzer(num_classes=2, feature_dim=512):
    model = FrequencyDomainInspector(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Now, let's create the training script for the frequency analyzer:


```python
import os
import argparse
from models.frequency_analyzer import create_frequency_analyzer
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Frequency Domain Analyzer Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/frequency_analyzer', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Create model
    model = create_frequency_analyzer(
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
```

Let's create the Biological Cues Detector:


```python
import torch
import torch.nn as nn
import cv2
import numpy as np
import face_recognition
import mediapipe as mp

class BiologicalCuesDetector(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.rppg_analyzer = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.micro_expression_net = nn.Sequential(
            nn.Linear(468 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def extract_facial_landmarks(self, x):
        batch_size = x.shape[0]
        landmark_features = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            img_rgb = (img * 255).astype(np.uint8)
            
            results = self.mp_face_mesh.process(img_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                coords = []
                for landmark in landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                # Ensure length is always 1404 (468 * 3)
                if len(coords) < 468 * 3:
                    coords += [0.0] * (468 * 3 - len(coords))
                elif len(coords) > 468 * 3:
                    coords = coords[:468 * 3]
                landmark_features.append(coords)
            else:
                landmark_features.append([0.0] * (468 * 3))
        
        return torch.FloatTensor(np.array(landmark_features)).to(x.device)
    
    def extract_rppg_signal(self, x):
        batch_size = x.shape[0]
        rppg_signals = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            
            face_locations = face_recognition.face_locations((img * 255).astype(np.uint8))
            if face_locations:
                top, right, bottom, left = face_locations[0]
                face_roi = img[top:bottom, left:right]
                
                if face_roi.size > 0:
                    rgb_signal = np.mean(face_roi, axis=(0, 1))
                    rppg_signals.append(rgb_signal)
                else:
                    rppg_signals.append([0.0, 0.0, 0.0])
            else:
                rppg_signals.append([0.0, 0.0, 0.0])
        
        rppg_tensor = torch.FloatTensor(np.array(rppg_signals)).to(x.device)
        return rppg_tensor.unsqueeze(2)  # [batch, 3, 1]
    
    def forward(self, x):
        landmark_features = self.extract_facial_landmarks(x)
        rppg_signal = self.extract_rppg_signal(x)
        
        micro_expr_features = self.micro_expression_net(landmark_features)
        rppg_features = self.rppg_analyzer(rppg_signal).flatten(1)
        
        combined = torch.cat([micro_expr_features, rppg_features], dim=1)
        features = self.fusion(combined)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'landmarks': landmark_features,
            'rppg': rppg_signal
        }

def create_biological_detector(num_classes=2, feature_dim=512):
    model = BiologicalCuesDetector(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let's create the training script for the biological detector:


```python
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
```

Let's create the Semantic Consistency GNN:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class SemanticConsistencyGNN(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.node_encoder = nn.Linear(3, 64)
        self.gnn_layers = nn.ModuleList([
            gnn.GCNConv(64, 128),
            gnn.GCNConv(128, 256),
            gnn.GCNConv(256, 128)
        ])
        self.global_pool = gnn.global_mean_pool
        
        self.fusion = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def create_face_graph(self, landmarks):
        batch_graphs = []
        
        for batch_idx in range(landmarks.shape[0]):
            landmark_coords = landmarks[batch_idx].view(-1, 3)
            
            # Create edges between nearby landmarks
            edges = []
            for i in range(len(landmark_coords)):
                for j in range(i+1, min(i+10, len(landmark_coords))):
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions for undirected graph
            
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=landmark_coords, edge_index=edge_index.to(landmarks.device))
            batch_graphs.append(graph)
        
        return batch_graphs
    
    def forward(self, landmarks):
        batch_graphs = self.create_face_graph(landmarks)
        batch_features = []
        
        for graph in batch_graphs:
            x = self.node_encoder(graph.x)
            
            for gnn_layer in self.gnn_layers:
                x = F.relu(gnn_layer(x, graph.edge_index))
            
            graph_feature = self.global_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            batch_features.append(graph_feature)
        
        batch_tensor = torch.stack(batch_features)
        features = self.fusion(batch_tensor)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'graph_features': batch_tensor
        }

def create_semantic_gnn(num_classes=2, feature_dim=512):
    model = SemanticConsistencyGNN(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let me try that again:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class SemanticConsistencyGNN(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.node_encoder = nn.Linear(3, 64)
        self.gnn_layers = nn.ModuleList([
            gnn.GCNConv(64, 128),
            gnn.GCNConv(128, 256),
            gnn.GCNConv(256, 128)
        ])
        self.global_pool = gnn.global_mean_pool
        
        self.fusion = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def create_face_graph(self, landmarks):
        batch_graphs = []
        
        for batch_idx in range(landmarks.shape[0]):
            landmark_coords = landmarks[batch_idx].view(-1, 3)
            
            # Create edges between nearby landmarks
            edges = []
            for i in range(len(landmark_coords)):
                for j in range(i+1, min(i+10, len(landmark_coords))):
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions for undirected graph
            
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=landmark_coords, edge_index=edge_index.to(landmarks.device))
            batch_graphs.append(graph)
        
        return batch_graphs
    
    def forward(self, landmarks):
        batch_graphs = self.create_face_graph(landmarks)
        batch_features = []
        
        for graph in batch_graphs:
            x = self.node_encoder(graph.x)
            
            for gnn_layer in self.gnn_layers:
                x = F.relu(gnn_layer(x, graph.edge_index))
            
            graph_feature = self.global_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            batch_features.append(graph_feature)
        
        batch_tensor = torch.stack(batch_features)
        features = self.fusion(batch_tensor)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'graph_features': batch_tensor
        }

def create_semantic_gnn(num_classes=2, feature_dim=512):
    model = SemanticConsistencyGNN(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let me try one more time with a different approach:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class SemanticConsistencyGNN(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.node_encoder = nn.Linear(3, 64)
        self.gnn_layers = nn.ModuleList([
            gnn.GCNConv(64, 128),
            gnn.GCNConv(128, 256),
            gnn.GCNConv(256, 128)
        ])
        self.global_pool = gnn.global_mean_pool
        
        self.fusion = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def create_face_graph(self, landmarks):
        batch_graphs = []
        
        for batch_idx in range(landmarks.shape[0]):
            landmark_coords = landmarks[batch_idx].view(-1, 3)
            
            # Create edges between nearby landmarks
            edges = []
            for i in range(len(landmark_coords)):
                for j in range(i+1, min(i+10, len(landmark_coords))):
                    edges.append([i, j])
                    edges.append([j, i])  # Add both directions for undirected graph
            
            if edges:
                edge_index = torch.tensor(edges).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            graph = Data(x=landmark_coords, edge_index=edge_index.to(landmarks.device))
            batch_graphs.append(graph)
        
        return batch_graphs
    
    def forward(self, landmarks):
        batch_graphs = self.create_face_graph(landmarks)
        batch_features = []
        
        for graph in batch_graphs:
            x = self.node_encoder(graph.x)
            
            for gnn_layer in self.gnn_layers:
                x = F.relu(gnn_layer(x, graph.edge_index))
            
            graph_feature = self.global_pool(x, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
            batch_features.append(graph_feature)
        
        batch_tensor = torch.stack(batch_features)
        features = self.fusion(batch_tensor)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'graph_features': batch_tensor
        }

def create_semantic_gnn(num_classes=2, feature_dim=512):
    model = SemanticConsistencyGNN(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let's create the training script for the semantic GNN:


```python
import os
import argparse
from models.semantic_gnn import create_semantic_gnn
from models.biological_detector import create_biological_detector
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer

class SemanticGNNTrainer(BaseTrainer):
    def __init__(self, model, landmark_extractor, save_dir='checkpoints/semantic_gnn', device=None):
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
    parser = argparse.ArgumentParser(description='Train Semantic Consistency GNN Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/semantic_gnn', help='Directory to save checkpoints')
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

    # Create GNN model
    model = create_semantic_gnn(
        num_classes=2,
        feature_dim=args.feature_dim
    )

    # Create trainer
    trainer = SemanticGNNTrainer(
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
```

Let's create the Geometry 3D Analyzer:


```python
import torch
import torch.nn as nn
import numpy as np

class Geometry3DAnalyzer(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.pose_estimator = nn.Sequential(
            nn.Linear(468 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(64 + 6, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features
        }
    
    def estimate_head_pose(self, landmarks):
        batch_size = landmarks.shape[0]
        pose_features = []
        
        for i in range(batch_size):
            landmark_3d = landmarks[i].view(-1, 3).cpu().numpy()
            
            if len(landmark_3d) >= 6:
                # Extract key facial landmarks
                nose_tip = landmark_3d[1]
                left_eye = landmark_3d[33]
                right_eye = landmark_3d[263]
                left_mouth = landmark_3d[61]
                right_mouth = landmark_3d[291]
                chin = landmark_3d[18]
                
                # Calculate pose features
                pose_vector = np.array([
                    nose_tip[0] - (left_eye[0] + right_eye[0]) / 2,  # Nose-eye alignment
                    nose_tip[1] - (left_eye[1] + right_eye[1]) / 2,  # Vertical alignment
                    nose_tip[2] - (left_eye[2] + right_eye[2]) / 2,  # Depth alignment
                    (right_eye[0] - left_eye[0]),  # Eye distance
                    (right_mouth[0] - left_mouth[0]),  # Mouth width
                    chin[1] - nose_tip[1]  # Face height ratio
                ])
            else:
                pose_vector = np.zeros(6)
            
            pose_features.append(pose_vector)
        
        return torch.FloatTensor(np.array(pose_features)).to(landmarks.device)
    
    def forward(self, landmarks):
        pose_features = self.estimate_head_pose(landmarks)
        landmark_features = self.pose_estimator(landmarks.view(landmarks.shape[0], -1))
        
        combined = torch.cat([landmark_features, pose_features], dim=1)
        features = self.fusion(combined)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'pose_features': pose_features,
            'landmark_features': landmark_features
        }

def create_geometry_analyzer(num_classes=2, feature_dim=512):
    model = Geometry3DAnalyzer(num_classes=num_classes, feature_dim=feature_dim)
    return model
```

Let's create the training script for the geometry analyzer:


```python
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
```

Let's create the Attention Maps component:


```python
import torch
import torch.nn as nn

class AttentionMaps(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512, input_channels=3):
        super().__init__()
        attn_channels = max(1, input_channels // 4)
        
        # Attention mechanism
        self.attention_conv = nn.Sequential(
            nn.Conv2d(input_channels, attn_channels, kernel_size=1),
            nn.BatchNorm2d(attn_channels),
            nn.ReLU(),
            nn.Conv2d(attn_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion[0].out_features,
            'input_channels': self.attention_conv[0].in_channels
        }
    
    def forward(self, x):
        # Generate attention maps
        attention_weights = self.attention_conv(x)
        
        # Apply attention
        attended_features = x * attention_weights
        
        # Extract features
        features = self.feature_extractor(attended_features)
        features = features.flatten(1)
        
        # Process features
        features = self.fusion(features)
        output = self.classifier(features)
        
        return {
            'output': output,
            'features': features,
            'attention_weights': attention_weights,
            'attended_features': attended_features
        }

def create_attention_maps(num_classes=2, feature_dim=512, input_channels=3):
    model = AttentionMaps(
        num_classes=num_classes,
        feature_dim=feature_dim,
        input_channels=input_channels
    )
    return model
```

Let's create the training script for the attention maps:


```python
import os
import argparse
from models.attention_maps import create_attention_maps
from dataset.dataset import create_data_loader
from trainers.base_trainer import BaseTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Attention Maps Component')
    parser.add_argument('--train_real_dir', type=str, required=True, help='Directory containing real training images')
    parser.add_argument('--train_fake_dir', type=str, required=True, help='Directory containing fake training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='checkpoints/attention_maps', help='Directory to save checkpoints')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--input_channels', type=int, default=3, help='Number of input channels')
    args = parser.parse_args()

    # Create data loader
    train_loader = create_data_loader(
        args.train_real_dir,
        args.train_fake_dir,
        batch_size=args.batch_size
    )

    # Create model
    model = create_attention_maps(
        num_classes=2,
        feature_dim=args.feature_dim,
        input_channels=args.input_channels
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
```

Finally, let's create the Dynamic Fusion Network that will combine all components:


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicFusionNetwork(nn.Module):
    def __init__(self, input_dims, num_classes=2, feature_dim=512):
        super().__init__()
        self.input_dims = input_dims
        total_dim = sum(input_dims)
        
        # Uncertainty estimation for each component
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(input_dims)),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def get_config(self):
        return {
            'num_classes': self.classifier[-1].out_features,
            'feature_dim': self.fusion_net[3].out_features,
            'input_dims': self.input_dims
        }
    
    def forward(self, features_list):
        # Ensure all features are [batch, C]
        features_list = [f.flatten(1) if f.dim() > 2 else f for f in features_list]
        combined_features = torch.cat(features_list, dim=1)
        
        # Estimate uncertainty weights for each component
        uncertainty_weights = self.uncertainty_estimator(combined_features)
        
        # Apply uncertainty-weighted fusion
        weighted_features = []
        start_idx = 0
        for i, features in enumerate(features_list):
            end_idx = start_idx + features.shape[1]
            weight = uncertainty_weights[:, i:i+1]
            weighted_features.append(features * weight)
            start_idx = end_idx
        
        final_features = torch.cat(weighted_features, dim=1)
        
        # Process fused features
        features = self.fusion_net(final_features)
        output = self.classifier(features)
        
        # Estimate confidence
        confidence = self.confidence_estimator(combined_features)
        
        return {
            'output': output,
            'features': features,
            'confidence': confidence,
            'uncertainty_weights': uncertainty_weights,
            'component_features': features_list
        }

def create_fusion_network(input_dims, num_classes=2, feature_dim=512):
    model = DynamicFusionNetwork(
        input_dims=input_dims,
        num_classes=num_classes,
        feature_dim=feature_dim
    )
    return model
```

Let's create the training script for the fusion network:


```python
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
```

I've now created all the necessary components and their training scripts. Here's a summary of what I've implemented:

1. Base Components:
   - `dataset/dataset.py` - Common dataset handling
   - `trainers/base_trainer.py` - Base training functionality

2. Individual Components:
   - Texture Analyzer (`models/texture_analyzer.py`)
   - Frequency Domain Inspector (`models/frequency_analyzer.py`)
   - Biological Cues Detector (`models/biological_detector.py`)
   - Semantic Consistency GNN (`models/semantic_gnn.py`)
   - Geometry 3D Analyzer (`models/geometry_analyzer.py`)
   - Attention Maps (`models/attention_maps.py`)
   - Dynamic Fusion Network (`models/fusion_network.py`)

3. Training Scripts:
   - `trainers/train_texture_analyzer.py`
   - `trainers/train_frequency_analyzer.py`
   - `trainers/train_biological_detector.py`
   - `trainers/train_semantic_gnn.py`
   - `trainers/train_geometry_analyzer.py`
   - `trainers/train_attention_maps.py`
   - `trainers/train_fusion_network.py`

To train the complete system, you would:

1. First train each component independently:
```bash
# Train texture analyzer
python trainers/train_texture_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train frequency analyzer
python trainers/train_frequency_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train biological detector
python trainers/train_biological_detector.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train semantic GNN (requires biological detector)
python trainers/train_semantic_gnn.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake --landmark_model_path checkpoints/biological_detector/best_model.pth

# Train geometry analyzer (requires biological detector)
python trainers/train_geometry_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake --landmark_model_path checkpoints/biological_detector/best_model.pth

# Train attention maps
python trainers/train_attention_maps.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake
```

2. Finally, train the fusion network using all pretrained components:
```bash
python trainers/train_fusion_network.py \
    --train_real_dir Dataset/Train/Real \
    --train_fake_dir Dataset/Train/Fake \
    --texture_model_path checkpoints/texture_analyzer/best_model.pth \
    --frequency_model_path checkpoints/frequency_analyzer/best_model.pth \
    --biological_model_path checkpoints/biological_detector/best_model.pth \
    --semantic_model_path checkpoints/semantic_gnn/best_model.pth \
    --geometry_model_path checkpoints/geometry_analyzer/best_model.pth \
    --attention_model_path checkpoints/attention_maps/best_model.pth
```

Each component can be trained independently and in parallel, and the fusion network will combine their features for the final prediction. The system is modular and each component can be improved or replaced independently.

Would you like me to explain any specific component in more detail or help you start training any particular component?

```shellscript
# Train texture analyzer
python trainers/train_texture_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train frequency analyzer
python trainers/train_frequency_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train biological detector
python trainers/train_biological_detector.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake

# Train semantic GNN (requires biological detector)
python trainers/train_semantic_gnn.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake --landmark_model_path checkpoints/biological_detector/best_model.pth

# Train geometry analyzer (requires biological detector)
python trainers/train_geometry_analyzer.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake --landmark_model_path checkpoints/biological_detector/best_model.pth

# Train attention maps
python trainers/train_attention_maps.py --train_real_dir Dataset/Train/Real --train_fake_dir Dataset/Train/Fake
```

```shellscript
python trainers/train_fusion_network.py \
    --train_real_dir Dataset/Train/Real \
    --train_fake_dir Dataset/Train/Fake \
    --texture_model_path checkpoints/texture_analyzer/best_model.pth \
    --frequency_model_path checkpoints/frequency_analyzer/best_model.pth \
    --biological_model_path checkpoints/biological_detector/best_model.pth \
    --semantic_model_path checkpoints/semantic_gnn/best_model.pth \
    --geometry_model_path checkpoints/geometry_analyzer/best_model.pth \
    --attention_model_path checkpoints/attention_maps/best_model.pth
```

