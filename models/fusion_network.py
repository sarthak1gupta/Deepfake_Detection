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