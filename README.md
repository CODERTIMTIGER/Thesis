# YOLOX SAR Target Detection System

A comprehensive MATLAB implementation of YOLOX object detection for Synthetic Aperture Radar (SAR) imagery using the Mix_MSTAR dataset. This system supports end-to-end detection, classification, and orientation estimation of military vehicles in SAR images.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

This project implements a state-of-the-art object detection system for SAR target recognition using YOLOX architecture. The system is designed specifically for the Mix_MSTAR dataset (640x640 images) and includes:

- **Detection**: Localize targets with bounding boxes
- **Classification**: Identify vehicle types (10 MSTAR classes)
- **Orientation Estimation**: Predict target orientation angles using Huber loss
- **End-to-end Training**: Single-stage YOLO approach for real-time performance

### Key Advantages
- **Anchor-free Detection**: YOLOX eliminates complex anchor box tuning
- **Decoupled Head**: Separate branches for classification and regression
- **SAR-Optimized**: Custom preprocessing for SAR imagery characteristics
- **Orientation Estimation**: Novel addition using Huber loss for robust angle prediction

## Features

### Core Capabilities
✅ YOLOX-based object detection (nano, tiny, small, medium, large, xlarge)
✅ Multi-class vehicle classification (10 MSTAR classes)
✅ Orientation estimation with custom Huber loss
✅ Complete IoU (CIoU) loss for accurate bounding box regression
✅ SAR-specific preprocessing (log transform, normalization)
✅ Comprehensive evaluation metrics (mAP, precision, recall, F1)
✅ Real-time inference on 640x640 images
✅ Pretrained model support (COCO transfer learning)
✅ Custom training with dlnetwork for full control
✅ Advanced visualization tools

### SAR-Specific Features
- Log transformation for dynamic range compression
- Speckle noise handling
- Aspect ratio preservation (critical for SAR)
- Single-channel to RGB conversion for pretrained models
- Orientation wrap-around handling

## Architecture

### YOLOX Model Structure

```
Input (640×640×3)
      ↓
┌─────────────────┐
│   BACKBONE      │  CSPDarknet
│  (Feature       │  - Focus layer
│   Extraction)   │  - CSP blocks (5 stages)
│                 │  - Multi-scale features
└─────────────────┘
      ↓
  P3, P4, P5 (80×80, 40×40, 20×20)
      ↓
┌─────────────────┐
│     NECK        │  PANet
│  (Feature       │  - Top-down pathway (FPN)
│   Fusion)       │  - Bottom-up augmentation
└─────────────────┘
      ↓
┌─────────────────┐
│     HEAD        │  Decoupled Head
│  (Detection)    │  - Classification branch
│                 │  - Regression branch (bbox)
│                 │  - Objectness branch
│                 │  - Orientation branch (CUSTOM)
└─────────────────┘
      ↓
Output: [BBox, Confidence, Class, Orientation]
```

### Loss Function

**Total Loss**:
```
L_total = λ_box * L_CIoU + λ_obj * L_obj + λ_cls * L_cls + λ_ori * L_Huber
```

**Components**:
1. **Bounding Box Loss (CIoU)**: Complete IoU with aspect ratio and center distance
2. **Objectness Loss**: Binary cross-entropy for object presence
3. **Classification Loss**: Cross-entropy for vehicle type
4. **Orientation Loss (Huber)**: Robust angle estimation with smooth gradients

**Huber Loss for Orientation**:
```matlab
L_Huber(e) = { 0.5 * e²           if |e| ≤ δ
             { δ(|e| - 0.5δ)      if |e| > δ
```
- Combines MSE (smooth gradients) and MAE (outlier robustness)
- Handles angular wrap-around: error ∈ [-180°, 180°]

## Requirements

### Software
- **MATLAB R2024b** or later
- **Computer Vision Toolbox**
- **Deep Learning Toolbox**
- **Image Processing Toolbox**
- **Parallel Computing Toolbox** (recommended for GPU training)

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **RAM**: 16GB+ for training
- **Storage**: 10GB+ for dataset and models

### Recommended Specifications
- NVIDIA RTX 3080/4080 or better
- 32GB RAM
- SSD storage for faster data loading

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/yolox-sar-detection.git
cd yolox-sar-detection
```

2. **Add to MATLAB path**:
```matlab
addpath(genpath('src'));
addpath('configs');
```

3. **Verify toolboxes**:
```matlab
% Check required toolboxes
ver('vision')           % Computer Vision Toolbox
ver('nnet')             % Deep Learning Toolbox
ver('images')           % Image Processing Toolbox
ver('parallel')         % Parallel Computing Toolbox
```

4. **Check GPU availability**:
```matlab
gpuDevice  % Should display GPU info if available
```

## Dataset Preparation

### Mix_MSTAR Dataset Structure

Organize your Mix_MSTAR dataset as follows:

```
data/
├── train/
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── val/
│   ├── image_1001.jpg
│   └── ...
├── test/
│   ├── image_2001.jpg
│   └── ...
└── annotations/
    ├── annotations.mat          % MATLAB format (recommended)
    ├── image_0001.txt          % YOLO format
    └── ...
```

### Annotation Format

**Option 1: MATLAB Table (Recommended)**
```matlab
% annotations.mat contains:
annotations = table(ImagePath, BBox, Label, Orientation);

% Where:
% ImagePath: Full path to image
% BBox: [x, y, width, height] (top-left corner format)
% Label: Vehicle class name
% Orientation: Angle in degrees [-180, 180]
```

**Option 2: YOLO Format (TXT)**
```
# Each line: class x_center y_center width height orientation
# Coordinates normalized to [0, 1]
0 0.5 0.5 0.1 0.15 45.0
```

### MSTAR Vehicle Classes
1. BMP2 - Infantry Fighting Vehicle
2. BTR70 - Armored Personnel Carrier
3. T72 - Main Battle Tank
4. BTR60 - Armored Personnel Carrier
5. BMP1 - Infantry Fighting Vehicle
6. 2S1 - Self-Propelled Howitzer
7. BRDM2 - Reconnaissance Vehicle
8. D7 - Bulldozer
9. T62 - Main Battle Tank
10. ZIL131 - Truck

## Configuration

Edit `configs/config.m` to customize training parameters:

### Key Configuration Parameters

```matlab
% Model selection
YOLOX_MODEL = 'small';              % 'nano', 'tiny', 'small', 'medium', 'large'
NUM_CLASSES = 10;                   % MSTAR vehicle classes

% Training hyperparameters
MAX_EPOCHS = 100;
MINI_BATCH_SIZE = 4;                % Adjust based on GPU memory
INITIAL_LEARNING_RATE = 1e-3;

% Loss weights
LAMBDA_BOX = 5.0;                   % Bounding box importance
LAMBDA_OBJ = 1.0;                   % Objectness importance
LAMBDA_CLS = 0.5;                   % Classification importance
LAMBDA_ORI = 0.3;                   % Orientation importance

% Orientation settings
DETECT_ORIENTATION = true;          % Enable/disable orientation
HUBER_DELTA = 1.0;                  % Huber loss threshold

% Data augmentation
USE_MOSAIC = true;                  % Mosaic augmentation
USE_MIXUP = true;                   % MixUp augmentation
RANDOM_ROTATION = [-10, 10];        % Degrees
```

## Usage

### Quick Start Example

```matlab
% 1. Load configuration
cfg = config();
cfg.printConfig();

% 2. Prepare data
[trainData, valData, testData] = prepareData('data/', 'data/annotations/', ...
    'IncludeOrientation', true, 'Split', [0.7, 0.15, 0.15]);

% 3. Setup YOLOX detector
detector = setupYOLOX('ModelSize', 'small', 'NumClasses', 10, ...
                      'Pretrained', true, 'CustomHead', true);

% 4. Train model
[trainedDetector, info] = trainYOLOX(trainData, valData, ...
                                     'SaveCheckpoints', true);

% 5. Evaluate
[metrics, results] = evaluateModel(trainedDetector, testData, 'Visualize', true);

% 6. Inference on new image
[bboxes, scores, labels, orientations] = inferenceYOLOX(trainedDetector, 'test_image.jpg', ...
                                                         'Visualize', true);
```

## Training

### Standard Training (Built-in YOLOX)

```matlab
% Prepare data
[trainData, valData, testData] = prepareData('data/', 'data/annotations/');

% Create standard YOLOX detector
detector = setupYOLOX('ModelSize', 'small', 'NumClasses', 10, ...
                      'Pretrained', true, 'NetworkType', 'standard');

% Train
[trainedDetector, info] = trainYOLOX(trainData, valData);

% Save model
save('models/yolox_sar_trained.mat', 'trainedDetector', 'info');
```

### Custom Training (with Orientation)

```matlab
% Setup custom YOLOX with orientation head
detector = setupYOLOX('ModelSize', 'small', 'CustomHead', true, ...
                      'NetworkType', 'custom');

% Train with custom loss
[trainedDetector, info] = trainYOLOX(trainData, valData, ...
                                     'Detector', detector, ...
                                     'SaveCheckpoints', true);
```

### Resume Training from Checkpoint

```matlab
[trainedDetector, info] = trainYOLOX(trainData, valData, ...
                                     'ResumeFrom', 'models/checkpoints/epoch_50.mat');
```

### Monitor Training

Training progress is automatically plotted with:
- Training loss curve
- Validation loss curve
- Component losses (bbox, objectness, classification, orientation)

You can also monitor metrics in real-time:
```matlab
% Access training history
plot(info.TrainingLoss);
hold on;
plot(info.ValidationLoss);
legend('Training', 'Validation');
xlabel('Epoch');
ylabel('Loss');
title('Training Progress');
```

## Evaluation

### Comprehensive Evaluation

```matlab
% Evaluate on test set
[metrics, results] = evaluateModel(trainedDetector, testData, ...
                                   'IoUThreshold', 0.5:0.05:0.95, ...
                                   'Visualize', true, ...
                                   'SaveResults', true);

% Display metrics
disp('=== Evaluation Results ===');
fprintf('mAP@0.5: %.2f%%\n', metrics.mAP.mAP_50 * 100);
fprintf('mAP@0.75: %.2f%%\n', metrics.mAP.mAP_75 * 100);
fprintf('mAP@[0.5:0.95]: %.2f%%\n', metrics.mAP.mAP_50_95 * 100);
fprintf('Precision: %.2f%%\n', metrics.Overall.Precision * 100);
fprintf('Recall: %.2f%%\n', metrics.Overall.Recall * 100);
fprintf('F1-Score: %.2f%%\n', metrics.Overall.F1 * 100);

% Orientation metrics (if available)
if isfield(metrics, 'Orientation')
    fprintf('\nOrientation Estimation:\n');
    fprintf('MAAE: %.2f degrees\n', metrics.Orientation.MAAE);
    fprintf('Accuracy (15°): %.2f%%\n', metrics.Orientation.Accuracy_15deg);
end
```

### Per-Class Performance

```matlab
% Analyze per-class metrics
classNames = config.CLASS_NAMES;
for i = 1:length(classNames)
    className = classNames{i};
    if isfield(metrics.PerClass, className)
        fprintf('%s:\n', className);
        fprintf('  Precision: %.2f%%\n', metrics.PerClass.(className).Precision * 100);
        fprintf('  Recall: %.2f%%\n', metrics.PerClass.(className).Recall * 100);
        fprintf('  F1: %.2f%%\n', metrics.PerClass.(className).F1 * 100);
    end
end
```

## Inference

### Single Image Inference

```matlab
% Load trained model
load('models/yolox_sar_trained.mat', 'trainedDetector');

% Run inference
[bboxes, scores, labels, orientations] = inferenceYOLOX(trainedDetector, ...
    'data/test/test_image.jpg', ...
    'ConfidenceThreshold', 0.3, ...
    'Visualize', true, ...
    'SaveResults', true);

% Display results
for i = 1:size(bboxes, 1)
    fprintf('Detection %d: %s (%.2f%%), Orientation: %.1f°\n', ...
            i, labels{i}, scores(i)*100, orientations(i));
end
```

### Batch Inference

```matlab
% Process multiple images
imageFiles = dir('data/test/*.jpg');
allResults = cell(length(imageFiles), 1);

for i = 1:length(imageFiles)
    imgPath = fullfile(imageFiles(i).folder, imageFiles(i).name);
    [bboxes, scores, labels, ori] = inferenceYOLOX(trainedDetector, imgPath, ...
                                                     'Visualize', false);
    allResults{i} = struct('BBoxes', bboxes, 'Scores', scores, ...
                          'Labels', labels, 'Orientations', ori);
end

% Save batch results
save('results/batch_inference.mat', 'allResults');
```

## Visualization

### Detection Visualization

```matlab
% Visualize detections with all annotations
img = imread('test_image.jpg');
visualizeDetections(img, bboxes, scores, labels, orientations, ...
                   'Title', 'SAR Target Detection', ...
                   'ShowOrientation', true, ...
                   'ShowConfidence', true, ...
                   'ColorMap', 'class');
```

### Ground Truth Comparison

```matlab
% Compare predictions with ground truth
visualizeGroundTruthComparison(img, gtBBoxes, gtLabels, predBBoxes, predLabels);
```

### Orientation Distribution

```matlab
% Visualize orientation distribution
visualizeOrientationDistribution(orientations, labels);
```

### Confidence Distribution

```matlab
% Visualize confidence score distribution
visualizeConfidenceDistribution(scores, labels);
```

## Project Structure

```
Thesis/
├── README.md                          # This file
├── configs/
│   └── config.m                       # Configuration parameters
├── src/
│   ├── prepareData.m                  # Data loading and preprocessing
│   ├── trainYOLOX.m                   # Training pipeline
│   ├── evaluateModel.m                # Evaluation metrics
│   ├── inferenceYOLOX.m               # Inference script
│   ├── models/
│   │   ├── setupYOLOX.m               # Model architecture setup
│   │   └── customLossFunctions.m      # Loss functions (CIoU, Huber, etc.)
│   └── utils/
│       └── visualizeDetections.m      # Visualization utilities
├── data/
│   ├── train/                         # Training images
│   ├── val/                           # Validation images
│   ├── test/                          # Test images
│   └── annotations/                   # Annotation files
├── models/
│   ├── checkpoints/                   # Training checkpoints
│   └── pretrained/                    # Pretrained models
└── results/
    ├── visualizations/                # Output visualizations
    └── metrics/                       # Evaluation results
```

## Performance Benchmarks

### Expected Performance on MSTAR

| Metric | YOLOX-Nano | YOLOX-Small | YOLOX-Medium |
|--------|------------|-------------|--------------|
| mAP@0.5 | ~85% | ~92% | ~95% |
| mAP@0.75 | ~70% | ~80% | ~85% |
| Inference (640×640) | 5ms | 15ms | 30ms |
| GPU Memory | 2GB | 4GB | 8GB |
| Parameters | 0.9M | 9M | 25M |

### Orientation Estimation

| Metric | Expected |
|--------|----------|
| MAAE (Mean Absolute Angular Error) | < 10° |
| Accuracy within 15° | > 85% |
| Accuracy within 30° | > 95% |

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `MINI_BATCH_SIZE` in config
   - Use smaller model (`nano` or `tiny`)
   - Clear workspace: `clear; clc;`

2. **GPU Not Detected**
   ```matlab
   gpuDevice(1);  % Select GPU
   reset(gpuDevice);  % Reset GPU
   ```

3. **Slow Training**
   - Ensure GPU is being used
   - Reduce image size or batch size
   - Use pretrained model for faster convergence

4. **Poor Detection Performance**
   - Check data quality and annotations
   - Increase training epochs
   - Adjust loss weights in config
   - Try data augmentation

## Advanced Topics

### Transfer Learning

```matlab
% Load pretrained COCO weights
detector = yoloxObjectDetector('small-coco');

% Fine-tune on MSTAR
options = trainingOptions('adam', 'InitialLearnRate', 1e-4, ...
                         'MaxEpochs', 50);
trainedDetector = trainYOLOXObjectDetector(trainData, detector, options);
```

### Model Optimization

```matlab
% Quantization for faster inference
quantizedNet = dlquantize(trainedDetector.Network);

% Model pruning
prunedNet = prune(trainedDetector.Network, 'Threshold', 0.01);
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{yolox_sar_2025,
  title={YOLOX for SAR Target Detection with Orientation Estimation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/yolox-sar-detection}}
}
```

### Related Papers

1. **YOLOX**: Ge, Z., Liu, S., Wang, F., Li, Z., & Sun, J. (2021). YOLOX: Exceeding YOLO Series in 2021. arXiv preprint arXiv:2107.08430.

2. **MSTAR Dataset**: Ross, T. D., Worrell, S. W., Velten, V. J., Mossing, J. C., & Bryant, M. L. (1998). Standard SAR ATR evaluation experiments using the MSTAR public release data set.

3. **CIoU Loss**: Zheng, Z., Wang, P., Liu, W., Li, J., Ye, R., & Ren, D. (2020). Distance-IoU loss: Faster and better learning for bounding box regression.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- MathWorks for Computer Vision and Deep Learning Toolboxes
- MSTAR dataset contributors
- YOLOX authors for the excellent architecture
- SAR research community

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: your.email@university.edu

---

**Last Updated**: November 2025
**Version**: 1.0.0
**Status**: Research Implementation
