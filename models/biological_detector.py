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