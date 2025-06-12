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