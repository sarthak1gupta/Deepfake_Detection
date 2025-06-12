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