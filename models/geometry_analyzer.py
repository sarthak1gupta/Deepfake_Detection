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