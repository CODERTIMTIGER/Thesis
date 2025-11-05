# Complete YOLO Architecture Guide for SAR Detection

## Table of Contents
1. [Introduction to YOLO](#introduction-to-yolo)
2. [YOLO Evolution](#yolo-evolution)
3. [YOLOX Architecture Deep Dive](#yolox-architecture-deep-dive)
4. [Core Components](#core-components)
5. [Training Strategy](#training-strategy)
6. [Loss Functions](#loss-functions)
7. [Data Augmentation](#data-augmentation)
8. [SAR-Specific Adaptations](#sar-specific-adaptations)
9. [Implementation Details](#implementation-details)
10. [Performance Optimization](#performance-optimization)

---

## Introduction to YOLO

### What is YOLO?

**YOLO (You Only Look Once)** is a family of single-stage object detection algorithms that treat detection as a regression problem. Unlike two-stage detectors (R-CNN family), YOLO processes the entire image in one pass through the network.

**Key Concept**: "You Only Look Once" - the network sees the entire image once and predicts bounding boxes and class probabilities directly.

### Why YOLO for SAR?

1. **Speed**: Real-time detection critical for military applications
2. **End-to-end**: Single network handles detection and classification
3. **Global context**: Sees entire image, reducing false positives
4. **Adaptability**: Can be modified for orientation estimation

---

## YOLO Evolution

### Timeline of YOLO Versions

```
YOLOv1 (2016) → YOLOv2 (2017) → YOLOv3 (2018) → YOLOv4 (2020)
                                          ↓
                                     YOLOv5 (2020)
                                          ↓
                                    YOLOX (2021) ← We use this!
                                          ↓
                            YOLO11/YOLOv11 (2024)
```

### Key Improvements in Each Version

#### YOLOv1 (2016)
- First single-shot detector
- Divides image into S×S grid
- Each grid cell predicts B bounding boxes
- **Limitation**: Struggles with small objects

#### YOLOv2 (2017) / YOLO9000
- **Batch Normalization**: Improved convergence
- **Anchor Boxes**: Predefined boxes at multiple scales
- **High-Resolution Classifier**: Better small object detection
- **Multi-Scale Training**: Robustness to different sizes

#### YOLOv3 (2018)
- **Feature Pyramid**: Detections at 3 scales
- **Darknet-53 Backbone**: Deeper network with residual connections
- **Better Small Object Detection**: Through multi-scale predictions

#### YOLOv5 (2020)
- **PyTorch Implementation**: More accessible
- **Auto-augmentation**: Mosaic, MixUp
- **CSPNet Backbone**: Cross-stage partial connections
- **Focus Layer**: Efficient downsampling

#### YOLOX (2021) ⭐ **Our Choice**
- **Anchor-Free**: Eliminates anchor box tuning
- **Decoupled Head**: Separate classification and regression branches
- **SimOTA**: Advanced label assignment
- **Strong Augmentations**: Mosaic + MixUp
- **Multiple Sizes**: Nano to XLarge variants

**Why YOLOX?**
- Best speed/accuracy trade-off
- Excellent MATLAB support
- Anchor-free simplifies SAR adaptation
- Easy to add custom heads (orientation)

#### YOLO11 (2024)
- **Latest version**: State-of-the-art performance
- **Improved backbone**: Enhanced feature extraction
- **Better small object detection**: Critical for SAR
- **Consideration**: Less mature MATLAB support

---

## YOLOX Architecture Deep Dive

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOLOX Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image (640×640×3)                                       │
│         ↓                                                       │
│  ┌──────────────┐                                              │
│  │   BACKBONE   │  Feature Extraction (CSPDarknet)            │
│  │              │  → Outputs: P3, P4, P5 feature maps         │
│  └──────────────┘                                              │
│         ↓                                                       │
│  ┌──────────────┐                                              │
│  │     NECK     │  Feature Fusion (PANet)                     │
│  │              │  → Combines multi-scale features            │
│  └──────────────┘                                              │
│         ↓                                                       │
│  ┌──────────────┐                                              │
│  │     HEAD     │  Detection (Decoupled)                      │
│  │              │  → Separate branches for cls/reg            │
│  └──────────────┘                                              │
│         ↓                                                       │
│  Predictions: [BBox, Objectness, Class, Orientation]          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Input Processing

**Input Requirements**:
- **Size**: 640×640 pixels (default, configurable)
- **Channels**: 3 (RGB or replicated grayscale)
- **Normalization**: Typically [0, 1] range
- **Format**: MATLAB format [H, W, C, B]

**For SAR Images**:
```matlab
% Original SAR: 640×640×1 (single channel)
sarImg = imread('mstar_image.jpg');

% Preprocessing
sarImg = im2double(sarImg);              % Convert to [0,1]
sarImg = log(1 + sarImg);                % Log transform
sarImg = normalize(sarImg);              % Normalize
sarImg = repmat(sarImg, [1, 1, 3]);     % Convert to RGB

% Now ready for YOLOX: 640×640×3
```

---

## Core Components

### 1. BACKBONE: Feature Extraction

#### CSPDarknet Architecture

**Purpose**: Extract hierarchical features from input image

**Structure**:
```
Input (640×640×3)
    ↓
Focus Layer (320×320×12 → 320×320×32)
    ↓
Stage 1: CSP Block (320×320×32 → 320×320×64)
    ↓
Stage 2: CSP Block (160×160×64 → 160×160×128)
    ↓
Stage 3: CSP Block (80×80×128 → 80×80×256) → P3 output
    ↓
Stage 4: CSP Block (40×40×256 → 40×40×512) → P4 output
    ↓
Stage 5: CSP Block (20×20×512 → 20×20×1024) → P5 output
```

#### Focus Layer

**What it does**: Space-to-depth transformation

**Operation**:
```
Input: 640×640×3

Step 1: Slice into 2×2 patches
  [0,0] → Patch 1
  [0,1] → Patch 2
  [1,0] → Patch 3
  [1,1] → Patch 4

Step 2: Concatenate patches
  320×320×12  (4 patches × 3 channels = 12 channels)

Step 3: 1×1 Conv to desired channels
  320×320×32
```

**Advantages**:
- No information loss (vs. pooling/striding)
- Reduces spatial dimensions while increasing channels
- Efficient computation

#### CSP Block (Cross Stage Partial)

**Purpose**: Enhance gradient flow and reduce computation

**Structure**:
```
Input
  ↓
Split into two paths:
  ↓                    ↓
Path A             Path B
  ↓                    ↓
Conv + BN        Dense Blocks
  ↓                    ↓
ReLU                ReLU
  ↓                    ↓
     Concatenate
           ↓
     Transition
           ↓
        Output
```

**Benefits**:
- **Gradient reuse**: Information flows through both paths
- **Reduced parameters**: Fewer computations than standard ResNet
- **Better accuracy**: Enhanced feature learning

**MATLAB Implementation Concept**:
```matlab
function output = cspBlock(input, numFilters)
    % Split input
    [path_a, path_b] = split(input, 2);  % Split along channel

    % Path A: Simple convolution
    path_a = convolution2dLayer(1, numFilters/2);
    path_a = batchNormalizationLayer;
    path_a = leakyReluLayer(0.1);

    % Path B: Dense blocks
    path_b = denseBlock(path_b, numFilters/2);

    % Concatenate
    output = depthConcatenationLayer([path_a, path_b]);

    % Transition
    output = convolution2dLayer(1, numFilters);
    output = batchNormalizationLayer;
end
```

### 2. NECK: Feature Fusion

#### PANet (Path Aggregation Network)

**Purpose**: Combine features from different scales for better detection

**Components**:
1. **FPN (Feature Pyramid Network)**: Top-down pathway
2. **Bottom-up Path Augmentation**: Enhance localization
3. **Adaptive Feature Pooling**: Multi-scale fusion

**Architecture**:
```
Backbone Outputs:
  P5 (20×20×1024)
  P4 (40×40×512)
  P3 (80×80×256)

Top-Down Pathway (FPN):
  P5 → Upsample → + → P4
  P4 → Upsample → + → P3

Bottom-Up Pathway:
  P3 → Downsample → + → P4
  P4 → Downsample → + → P5

Final Output:
  Enhanced P3, P4, P5
```

**Why Multi-Scale?**
- **P3 (80×80)**: Detects small objects (distant vehicles)
- **P4 (40×40)**: Detects medium objects
- **P5 (20×20)**: Detects large objects (close vehicles)

**For SAR**:
- MSTAR vehicles are typically 30-80 pixels
- P3 and P4 are most important
- Can adjust detection scales based on expected target sizes

#### SPP (Spatial Pyramid Pooling)

**Purpose**: Increase receptive field without losing information

**Operation**:
```
Input: 20×20×1024
  ↓
Apply max pooling with different kernel sizes:
  - 5×5 pool → 20×20×1024
  - 9×9 pool → 20×20×1024
  - 13×13 pool → 20×20×1024
  ↓
Concatenate:
  20×20×4096
  ↓
1×1 Conv:
  20×20×1024
```

**Benefits**:
- Captures multi-scale context
- Fixed output size regardless of input
- Helps detect objects at various scales

### 3. HEAD: Detection

#### Decoupled Head Architecture

**Traditional YOLO Head**:
```
Feature Map → Conv → Output [bbox + cls + obj]
  - Single branch for all tasks
  - Task conflict: classification vs. localization
```

**Decoupled Head (YOLOX)**:
```
Feature Map
    ↓
  Split
    ↓         ↓         ↓
  Cls       Reg       Obj
  Branch   Branch    Branch
    ↓         ↓         ↓
  Conv      Conv      Conv
    ↓         ↓         ↓
  Conv      Conv      Conv
    ↓         ↓         ↓
  Classes  [x,y,w,h]  Score
```

**Our Custom Head (with Orientation)**:
```
Feature Map
    ↓
  Split
    ↓         ↓         ↓         ↓
  Cls       Reg       Obj       Ori
  Branch   Branch    Branch    Branch
    ↓         ↓         ↓         ↓
  Conv      Conv      Conv      Conv
    ↓         ↓         ↓         ↓
  Conv      Conv      Conv      Conv
    ↓         ↓         ↓         ↓
  10        [x,y,w,h]  1        1
  classes   bbox      conf     angle
```

**Benefits of Decoupling**:
- **Better convergence**: Each branch optimizes independently
- **Higher accuracy**: Reduces task conflict
- **Easier to extend**: Add custom branches (orientation)
- **Faster training**: More efficient gradient flow

#### Anchor-Free Detection

**Traditional (Anchor-Based)**:
```
Predefined anchor boxes: [30×30, 50×50, 80×80, ...]
For each anchor:
  - Predict offsets: Δx, Δy, Δw, Δh
  - Predict objectness
  - Predict class

Problem:
  - Many hyperparameters (anchor sizes, ratios)
  - Dataset-specific tuning needed
  - Difficult for SAR (varying target sizes)
```

**YOLOX (Anchor-Free)**:
```
For each grid cell:
  - Predict center: (x, y) directly
  - Predict size: (w, h) directly
  - Predict objectness
  - Predict class

Advantages:
  - Fewer hyperparameters
  - Generalizes better
  - Simpler for SAR adaptation
```

**Prediction Format**:
```
Per grid cell prediction:
  [x_offset, y_offset, w, h, objectness, class_1, ..., class_N, orientation]

Conversion to image coordinates:
  x_img = (x_offset + grid_x) × stride
  y_img = (y_offset + grid_y) × stride
  w_img = exp(w) × stride
  h_img = exp(h) × stride
```

---

## Training Strategy

### Label Assignment: SimOTA

**Problem**: How to assign ground truth to predictions?

**Traditional**: IoU-based matching (Hungarian algorithm)

**SimOTA (Simplified Optimal Transport Assignment)**:

**Concept**: Find optimal assignment minimizing total cost

**Cost Function**:
```
Cost = λ_cls × L_cls + λ_reg × L_reg + λ_iou × L_iou
```

**Advantages**:
- **Dynamic**: Adapts to each image
- **Multi-positive**: Multiple predictions can match one GT
- **Better gradient**: More positive samples for training

**For SAR**:
- Helps with small targets
- Reduces class imbalance issues
- Better handles overlapping vehicles

### Training Schedule

**Warm-up Phase (5-10 epochs)**:
```
Learning rate: 0 → Initial LR (linear)
Purpose: Stabilize training at start
```

**Main Training (50-90 epochs)**:
```
Learning rate: Initial LR → LR/10 (step decay)
Decay at: Epochs 60, 80
```

**Fine-tuning (Optional)**:
```
Learning rate: Very low (1e-5)
Purpose: Final refinement
```

**For MSTAR** (typical):
```
Total epochs: 100
Initial LR: 1e-3
Batch size: 4-8 (depends on GPU)
Optimizer: Adam
Warm-up: 5 epochs
Decay: 0.1 at epochs 70, 90
```

### Multi-Scale Training

**Concept**: Train on different image sizes

**YOLOX Approach**:
```
Every 10 epochs, randomly choose size:
  [480, 512, 544, 576, 608, 640, 672, 704, 736]

Benefits:
  - Robustness to scale changes
  - Better generalization
  - Handles varying target distances
```

**For SAR**:
```
Fixed size: 640×640
Reason: MSTAR images are consistent size
Alternative: Use zoom augmentation instead
```

---

## Loss Functions

### Complete Loss Function

```
L_total = λ_box × L_box + λ_obj × L_obj + λ_cls × L_cls + λ_ori × L_ori
```

### 1. Bounding Box Loss: CIoU

**Why not MSE?**
- Not scale-invariant
- Doesn't consider IoU
- Poor gradient for non-overlapping boxes

**Evolution of IoU Losses**:

#### IoU Loss
```
L_IoU = 1 - IoU(pred, gt)

IoU = Area(pred ∩ gt) / Area(pred ∪ gt)
```

#### GIoU (Generalized IoU)
```
L_GIoU = 1 - GIoU

GIoU = IoU - |C \ (A ∪ B)| / |C|

Where C = smallest box enclosing pred and gt
```
- Handles non-overlapping boxes
- Provides gradient even when IoU=0

#### DIoU (Distance IoU)
```
L_DIoU = 1 - IoU + ρ²(b_pred, b_gt) / c²

Where:
  ρ = Euclidean distance between box centers
  c = diagonal of smallest enclosing box
```
- Considers center distance
- Faster convergence than GIoU

#### CIoU (Complete IoU) ⭐ **Best for YOLOX**
```
L_CIoU = 1 - IoU + ρ²/c² + αν

Where:
  α = ν / (1 - IoU + ν)
  ν = (4/π²) × (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
```

**Advantages**:
- Considers **overlap** (IoU term)
- Considers **center distance** (ρ²/c² term)
- Considers **aspect ratio** (αν term)
- **Critical for SAR**: Preserves vehicle aspect ratio

**MATLAB Implementation**:
```matlab
function loss = ciouLoss(predBoxes, gtBoxes)
    % Calculate IoU
    iou = calculateIoU(predBoxes, gtBoxes);

    % Calculate center distance
    centerPred = [predBoxes(:,1) + predBoxes(:,3)/2, ...
                  predBoxes(:,2) + predBoxes(:,4)/2];
    centerGT = [gtBoxes(:,1) + gtBoxes(:,3)/2, ...
                gtBoxes(:,2) + gtBoxes(:,4)/2];
    centerDist = sum((centerPred - centerGT).^2, 2);

    % Calculate diagonal of enclosing box
    x1_enc = min(predBoxes(:,1), gtBoxes(:,1));
    y1_enc = min(predBoxes(:,2), gtBoxes(:,2));
    x2_enc = max(predBoxes(:,1)+predBoxes(:,3), ...
                 gtBoxes(:,1)+gtBoxes(:,3));
    y2_enc = max(predBoxes(:,2)+predBoxes(:,4), ...
                 gtBoxes(:,2)+gtBoxes(:,4));
    diagonal = (x2_enc - x1_enc).^2 + (y2_enc - y1_enc).^2;

    % Aspect ratio consistency
    v = (4/pi^2) * (atan(gtBoxes(:,4)./gtBoxes(:,3)) - ...
                    atan(predBoxes(:,4)./predBoxes(:,3))).^2;
    alpha = v ./ (1 - iou + v + eps);

    % CIoU loss
    loss = mean(1 - iou + centerDist./diagonal + alpha.*v);
end
```

### 2. Objectness Loss

**Purpose**: Predict if grid cell contains object

**Binary Cross-Entropy**:
```
L_obj = -[y × log(p) + (1-y) × log(1-p)]

Where:
  y = 1 if object present, 0 otherwise
  p = predicted objectness score
```

**Focal Loss Variant** (optional, for imbalance):
```
L_focal = -α(1-p)^γ × y × log(p)

Where:
  α = 0.25 (balance factor)
  γ = 2.0 (focusing parameter)
```

### 3. Classification Loss

**Cross-Entropy Loss**:
```
L_cls = -Σ y_i × log(p_i)

Where:
  y_i = one-hot encoded ground truth
  p_i = predicted class probabilities (after softmax)
```

**For MSTAR**:
```
10 classes: [BMP2, BTR70, T72, BTR60, BMP1, 2S1, BRDM2, D7, T62, ZIL131]

Example:
  Ground truth: T72 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  Prediction: [0.05, 0.10, 0.70, 0.05, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01]
  Loss = -(0×log(0.05) + 0×log(0.10) + 1×log(0.70) + ...)
       = -log(0.70) = 0.357
```

### 4. Orientation Loss: Huber

**Why Huber Loss?**

**MSE (Mean Squared Error)**:
```
L_MSE = (y - ŷ)²
```
- Smooth gradients
- Very sensitive to outliers
- Can explode with large errors

**MAE (Mean Absolute Error)**:
```
L_MAE = |y - ŷ|
```
- Robust to outliers
- Not smooth at error=0
- Slow convergence

**Huber Loss** ⭐ **Best of both**:
```
L_Huber = {
    0.5 × e²           if |e| ≤ δ
    δ(|e| - 0.5δ)      if |e| > δ
}

Where:
  e = error (with angle wrapping)
  δ = threshold (typically 1.0)
```

**Advantages**:
- **Smooth gradients** for small errors (like MSE)
- **Robust** to outliers (like MAE)
- **Perfect for orientation**: handles occasional large errors

**Angle Wrapping**:
```matlab
function wrappedError = wrapAngle(error)
    % Wrap to [-180, 180]
    wrappedError = mod(error + 180, 360) - 180;

    % Example:
    % Predicted: 350°, Ground truth: 10°
    % Naive error: 350 - 10 = 340°
    % Wrapped error: min(340, 360-340) = 20° ✓
end
```

**Complete Huber Implementation**:
```matlab
function loss = orientationHuberLoss(predAngles, gtAngles, delta)
    % Calculate error with wrapping
    error = predAngles - gtAngles;
    error = mod(error + 180, 360) - 180;  % Wrap to [-180, 180]

    absError = abs(error);

    % Huber loss
    isSmall = absError <= delta;
    loss = zeros(size(error));
    loss(isSmall) = 0.5 * error(isSmall).^2;
    loss(~isSmall) = delta * (absError(~isSmall) - 0.5*delta);

    loss = mean(loss);
end
```

### Loss Weight Selection

**Default Weights**:
```
λ_box = 5.0    % Most important
λ_obj = 1.0    % Standard
λ_cls = 0.5    % Less important than localization
λ_ori = 0.3    % Moderate importance
```

**Tuning Strategy**:
1. Start with standard weights
2. Monitor individual losses
3. If one loss dominates, reduce its weight
4. If one task performs poorly, increase its weight

**For SAR**:
- If localization is poor: Increase λ_box
- If many false positives: Increase λ_obj
- If classification errors: Increase λ_cls
- If orientation inaccurate: Increase λ_ori

---

## Data Augmentation

### Why Augmentation for SAR?

1. **Limited data**: MSTAR is relatively small
2. **Variation**: Simulate different conditions
3. **Robustness**: Handle real-world scenarios
4. **Overfitting prevention**: Regularization effect

### Standard Augmentations

#### Geometric Augmentations

**Rotation**:
```matlab
% Small rotations to preserve SAR characteristics
img = imrotate(img, randi([-10, 10]));
```
- Keep small (±10°) to maintain target appearance
- Larger rotations can distort SAR signatures

**Translation**:
```matlab
% Random shift
tx = randi([-20, 20]);
ty = randi([-20, 20]);
img = imtranslate(img, [tx, ty]);
```
- Helps network learn position invariance

**Horizontal Flip**:
```matlab
if rand > 0.5
    img = fliplr(img);
end
```
- Valid for SAR (left-right symmetry)

**Vertical Flip** (use with caution):
```matlab
% Less common for SAR
if rand > 0.8  % Use sparingly
    img = flipud(img);
end
```

#### Intensity Augmentations

**Brightness Jitter**:
```matlab
% Simulate different lighting/SAR conditions
factor = 0.8 + 0.4*rand;  % Range [0.8, 1.2]
img = img * factor;
img = max(0, min(1, img));
```

**Contrast Adjustment**:
```matlab
% Enhance or reduce contrast
factor = 0.8 + 0.4*rand;
mean_val = mean(img(:));
img = (img - mean_val) * factor + mean_val;
```

**Gaussian Noise** (simulate speckle):
```matlab
% Add noise to simulate SAR speckle
noise = 0.01 * randn(size(img));
img = img + noise;
```

### Advanced Augmentations (YOLOX-specific)

#### Mosaic Augmentation

**Purpose**: Combine 4 images into one, enhancing small object detection

**Process**:
```
┌─────────┬─────────┐
│ Image 1 │ Image 2 │
├─────────┼─────────┤
│ Image 3 │ Image 4 │
└─────────┴─────────┘
```

**Implementation**:
```matlab
function mosaicImg = createMosaic(img1, img2, img3, img4, bboxes)
    % Randomly choose split point
    splitX = randi([320-50, 320+50]);
    splitY = randi([320-50, 320+50]);

    % Create empty 640×640 canvas
    mosaicImg = zeros(640, 640, 3);

    % Place images
    mosaicImg(1:splitY, 1:splitX, :) = imresize(img1, [splitY, splitX]);
    mosaicImg(1:splitY, splitX+1:end, :) = imresize(img2, [splitY, 640-splitX]);
    mosaicImg(splitY+1:end, 1:splitX, :) = imresize(img3, [640-splitY, splitX]);
    mosaicImg(splitY+1:end, splitX+1:end, :) = imresize(img4, [640-splitY, 640-splitX]);

    % Adjust bounding boxes accordingly
    % (transform bbox coordinates to mosaic space)
end
```

**Benefits**:
- Richer context per image
- Batch normalization more effective
- Helps learn small objects

**For SAR**:
- Useful if targets are small
- May need adjustment for preservation of target aspect ratio

#### MixUp Augmentation

**Purpose**: Blend two images and their labels

**Formula**:
```
x_mix = λ × x1 + (1-λ) × x2
y_mix = λ × y1 + (1-λ) × y2

Where λ ~ Beta(α, α), typically α=1.5
```

**Implementation**:
```matlab
function [mixImg, mixLabel] = mixup(img1, label1, img2, label2)
    % Sample mixing coefficient
    lambda = betarnd(1.5, 1.5);

    % Mix images
    mixImg = lambda * img1 + (1 - lambda) * img2;

    % Mix labels (soft labels)
    mixLabel = lambda * label1 + (1 - lambda) * label2;
end
```

**Benefits**:
- Improved generalization
- Regularization effect
- Smoother decision boundaries

**For SAR** (use with caution):
- May blur SAR signatures
- Use lower mixing probability (10-15%)

### SAR-Specific Augmentations

#### Log Transform Variation
```matlab
% Vary log transform strength
alpha = 0.8 + 0.4*rand;  % [0.8, 1.2]
img = log(1 + alpha * img);
```

#### Simulated Speckle Noise
```matlab
% Multiplicative noise (SAR characteristic)
speckle = 1 + 0.1 * randn(size(img));
img = img .* speckle;
```

#### Aspect Ratio Preservation
```matlab
% CRITICAL: Never use non-uniform scaling
% Bad: img = imresize(img, [h, w]);  % Different h, w
% Good: Pad to square, then resize
maxDim = max(size(img, 1), size(img, 2));
imgPadded = padarray(img, [maxDim-size(img,1), maxDim-size(img,2)], 'post');
imgResized = imresize(imgPadded, [640, 640]);
```

---

## SAR-Specific Adaptations

### Challenges in SAR Imagery

1. **Speckle Noise**: Multiplicative noise inherent to SAR
2. **Single Channel**: Lack of color information
3. **Small Targets**: Vehicles are often 30-80 pixels
4. **Aspect Ratio Critical**: Vehicle dimensions matter for classification
5. **Orientation Important**: Target heading is valuable
6. **Limited Training Data**: MSTAR is relatively small

### Adaptations

#### Preprocessing Pipeline
```matlab
function processed = preprocessSAR(img)
    % 1. Load and convert to double
    img = im2double(img);

    % 2. Log transform (compress dynamic range)
    img = log(1 + img);

    % 3. Normalize (zero mean, unit variance)
    imgMean = mean(img(:));
    imgStd = std(img(:));
    img = (img - imgMean) / (imgStd + eps);

    % 4. Convert to 3-channel (for pretrained models)
    img = repmat(img, [1, 1, 3]);

    % 5. Resize to YOLOX input size
    img = imresize(img, [640, 640]);

    processed = img;
end
```

#### Backbone Adaptation

**Option 1: Use pretrained COCO weights**
```
Advantages:
  - Faster convergence
  - Better generalization
  - Proven feature extraction

Consideration:
  - COCO is RGB natural images
  - Need fine-tuning for SAR
```

**Option 2: Train from scratch**
```
Advantages:
  - Learns SAR-specific features
  - No RGB bias

Disadvantages:
  - Requires more data
  - Slower convergence
  - Risk of overfitting
```

**Recommended: Hybrid approach**
1. Start with COCO pretrained backbone
2. Freeze backbone initially
3. Train head on MSTAR
4. Unfreeze backbone
5. Fine-tune end-to-end with low learning rate

#### Network Adjustments

**Input Layer**:
```matlab
% Accept single channel input
inputLayer = imageInputLayer([640, 640, 1], 'Normalization', 'none');

% OR convert to 3-channel in preprocessing (recommended)
inputLayer = imageInputLayer([640, 640, 3], 'Normalization', 'none');
```

**First Conv Layer**:
```matlab
% Option: Modify first conv to accept 1 channel
% Load pretrained weights
pretrained = load('yolox_coco_weights.mat');

% Average RGB weights to grayscale
weights_rgb = pretrained.conv1.Weights;  % [K, K, 3, F]
weights_gray = mean(weights_rgb, 3);      % [K, K, 1, F]

% Assign to new network
net.Layers(1).Weights = weights_gray;
```

---

## Implementation Details

### dlnetwork vs. yoloxObjectDetector

#### yoloxObjectDetector (High-level)

**Pros**:
- Easy to use
- Built-in training functions
- Automatic optimization
- Good documentation

**Cons**:
- Less flexible
- Hard to add custom heads
- Limited control over loss function

**Use when**:
- Standard YOLOX is sufficient
- No orientation needed
- Quick prototyping

#### dlnetwork (Low-level)

**Pros**:
- Full control over architecture
- Custom heads (orientation)
- Custom loss functions
- Custom training loop

**Cons**:
- More complex
- Manual training loop
- Need to implement optimizers

**Use when**:
- Custom requirements (orientation)
- Research purposes
- Need full control

### Training Loop with dlnetwork

```matlab
function [net, info] = customTrainLoop(net, trainData, valData, cfg)
    % Initialize optimizer state
    avgGrad = [];
    avgGradSq = [];
    iteration = 0;

    % Training loop
    for epoch = 1:cfg.MAX_EPOCHS
        % Shuffle data
        trainData = shuffle(trainData);

        % Mini-batch loop
        while hasdata(trainData)
            iteration = iteration + 1;

            % Get batch
            [X, targets] = read(trainData);

            % Convert to dlarray
            X = dlarray(X, 'SSCB');
            if canUseGPU
                X = gpuArray(X);
            end

            % Compute loss and gradients
            [loss, gradients] = dlfeval(@modelLoss, net, X, targets, cfg);

            % Update parameters (Adam optimizer)
            [net, avgGrad, avgGradSq] = adamUpdate(net, gradients, ...
                avgGrad, avgGradSq, iteration, cfg.LEARNING_RATE);

            % Log progress
            if mod(iteration, 10) == 0
                fprintf('Epoch %d, Iter %d, Loss: %.4f\n', ...
                        epoch, iteration, extractdata(loss));
            end
        end

        % Validation
        valLoss = validateModel(net, valData, cfg);
        fprintf('Epoch %d, Val Loss: %.4f\n', epoch, valLoss);

        % Save checkpoint
        if mod(epoch, 10) == 0
            save(sprintf('checkpoint_epoch_%d.mat', epoch), 'net');
        end
    end
end
```

### Custom Loss Function

```matlab
function [totalLoss, gradients, lossComponents] = modelLoss(net, X, targets, cfg)
    % Forward pass
    Y = forward(net, X);

    % Parse predictions
    numClasses = cfg.NUM_CLASSES;
    bboxPred = Y(:,:,:,1:4);
    objPred = Y(:,:,:,5);
    clsPred = Y(:,:,:,6:5+numClasses);
    oriPred = Y(:,:,:,6+numClasses);

    % Compute individual losses
    lossBBox = ciouLoss(bboxPred, targets.BBox);
    lossObj = binaryCrossEntropy(objPred, targets.Objectness);
    lossCls = crossEntropy(clsPred, targets.Classes);
    lossOri = huberLoss(oriPred, targets.Orientation, 1.0);

    % Weighted combination
    totalLoss = cfg.LAMBDA_BOX * lossBBox + ...
                cfg.LAMBDA_OBJ * lossObj + ...
                cfg.LAMBDA_CLS * lossCls + ...
                cfg.LAMBDA_ORI * lossOri;

    % Compute gradients
    gradients = dlgradient(totalLoss, net.Learnables);

    % Store components for logging
    lossComponents.BBox = extractdata(lossBBox);
    lossComponents.Objectness = extractdata(lossObj);
    lossComponents.Classification = extractdata(lossCls);
    lossComponents.Orientation = extractdata(lossOri);
end
```

---

## Performance Optimization

### GPU Acceleration

```matlab
% Check GPU availability
gpuDevice

% Move data to GPU
X = gpuArray(X);

% Train on GPU
options = trainingOptions('adam', 'ExecutionEnvironment', 'gpu');

% Clear GPU memory
reset(gpuDevice);
```

### Batch Size Optimization

**Trade-off**:
- **Larger batch**: Faster training, more GPU memory, better gradients
- **Smaller batch**: Less memory, noisier gradients, may generalize better

**Finding optimal**:
```matlab
batchSizes = [2, 4, 8, 16];
for bs = batchSizes
    try
        % Try training with this batch size
        trainModel(trainData, 'BatchSize', bs);
        fprintf('Batch size %d works\n', bs);
    catch
        fprintf('Batch size %d: Out of memory\n', bs);
        break;
    end
end
```

**For MSTAR**:
- YOLOX-Nano: Batch size 16-32
- YOLOX-Small: Batch size 8-16
- YOLOX-Medium: Batch size 4-8
- With 8GB GPU

### Mixed Precision Training

```matlab
% Use half-precision (FP16) for speed
X = half(X);
net = half(net);

% Still compute loss in FP32 for stability
loss = single(computeLoss(net, X));
```

### Data Loading Optimization

```matlab
% Use parallel workers
trainData = transform(trainData, @preprocess, 'IncludeInfo', false);
trainData.NumWorkers = 4;  % Parallel preprocessing

% Prefetch data
trainData.Prefetch = true;
```

---

## Summary

### YOLOX for SAR: Key Takeaways

1. **Architecture**: Backbone → Neck → Head
2. **Anchor-Free**: Simpler, better for SAR
3. **Decoupled Head**: Separate classification and regression
4. **CIoU Loss**: Best for bounding box regression
5. **Huber Loss**: Ideal for orientation estimation
6. **Augmentation**: Mosaic + MixUp for robustness
7. **SAR Preprocessing**: Log transform + normalization
8. **Multi-Scale**: P3, P4, P5 for different target sizes

### Next Steps

1. **Understand**: Review this guide thoroughly
2. **Configure**: Set parameters in `configs/config.m`
3. **Prepare Data**: Organize MSTAR dataset
4. **Train**: Start with small model, then scale up
5. **Evaluate**: Check mAP, orientation accuracy
6. **Iterate**: Adjust hyperparameters based on results

---

**This completes the comprehensive YOLO architecture guide!**

For questions or issues, refer to the main README.md or open an issue on GitHub.
