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