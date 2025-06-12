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