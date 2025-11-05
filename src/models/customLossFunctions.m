%% CUSTOMLOSSFUNCTIONS - Custom loss functions for YOLOX SAR detection
%
% This file contains various loss functions for training YOLOX with
% orientation estimation:
%   - Bounding box losses (IoU, GIoU, DIoU, CIoU)
%   - Classification loss (Cross-Entropy, Focal Loss)
%   - Objectness loss (Binary Cross-Entropy)
%   - Orientation loss (Huber Loss, Circular Loss)
%
% Author: SAR Detection System
% Date: 2025-11-05

%% Complete YOLOX Loss Function
function [totalLoss, gradients, lossComponents] = yoloxLoss(net, X, Y, targets, cfg)
    % YOLOXLOSS - Complete multi-task loss function for YOLOX
    %
    % Inputs:
    %   net - dlnetwork object
    %   X - Input images (dlarray)
    %   Y - Ground truth targets (struct with BBox, Labels, Orientation)
    %   targets - Processed targets aligned with predictions
    %   cfg - Configuration object
    %
    % Outputs:
    %   totalLoss - Combined loss value
    %   gradients - Gradients for network update
    %   lossComponents - Struct with individual loss values
    %
    % Total Loss:
    %   L_total = λ_box*L_box + λ_obj*L_obj + λ_cls*L_cls + λ_ori*L_ori

    % Forward pass
    predictions = forward(net, X);

    % Parse predictions
    % Predictions format: [batch, H, W, features]
    % features = [x, y, w, h, objectness, class1, ..., classN, orientation]

    numClasses = cfg.NUM_CLASSES;
    bboxPred = predictions(:, :, :, 1:4);               % Bounding box [x,y,w,h]
    objPred = predictions(:, :, :, 5);                  % Objectness score
    clsPred = predictions(:, :, :, 6:5+numClasses);     % Class probabilities

    if cfg.DETECT_ORIENTATION
        oriPred = predictions(:, :, :, 6+numClasses);   % Orientation angle
    end

    % Extract ground truth
    bboxTarget = targets.BBox;
    objTarget = targets.Objectness;
    clsTarget = targets.Classes;

    if cfg.DETECT_ORIENTATION
        oriTarget = targets.Orientation;
    end

    % Compute individual losses
    lossBBox = boundingBoxLoss(bboxPred, bboxTarget, cfg.IOU_LOSS_TYPE);
    lossObj = objectnessLoss(objPred, objTarget);
    lossCls = classificationLoss(clsPred, clsTarget);

    if cfg.DETECT_ORIENTATION
        lossOri = orientationLoss(oriPred, oriTarget, cfg.HUBER_DELTA);
    else
        lossOri = dlarray(0);
    end

    % Weighted combination
    totalLoss = cfg.LAMBDA_BOX * lossBBox + ...
                cfg.LAMBDA_OBJ * lossObj + ...
                cfg.LAMBDA_CLS * lossCls + ...
                cfg.LAMBDA_ORI * lossOri;

    % Compute gradients
    gradients = dlgradient(totalLoss, net.Learnables);

    % Return loss components for monitoring
    lossComponents.Total = extractdata(totalLoss);
    lossComponents.BBox = extractdata(lossBBox);
    lossComponents.Objectness = extractdata(lossObj);
    lossComponents.Classification = extractdata(lossCls);
    lossComponents.Orientation = extractdata(lossOri);
end

%% Bounding Box Losses

function loss = boundingBoxLoss(pred, target, lossType)
    % BOUNDINGBOXLOSS - Compute bounding box regression loss
    %
    % Inputs:
    %   pred - Predicted boxes [x, y, w, h]
    %   target - Ground truth boxes [x, y, w, h]
    %   lossType - 'iou', 'giou', 'diou', 'ciou'
    %
    % Output:
    %   loss - Scalar loss value

    switch lower(lossType)
        case 'iou'
            loss = iouLoss(pred, target);
        case 'giou'
            loss = giouLoss(pred, target);
        case 'diou'
            loss = diouLoss(pred, target);
        case 'ciou'
            loss = ciouLoss(pred, target);
        otherwise
            error('Unknown IoU loss type: %s', lossType);
    end
end

function loss = iouLoss(pred, target)
    % IoU Loss: L = 1 - IoU
    iou = calculateIoU(pred, target);
    loss = mean(1 - iou, 'all');
end

function loss = giouLoss(pred, target)
    % GIoU Loss: Generalized IoU
    % L = 1 - GIoU
    % GIoU = IoU - |C \ (A ∪ B)| / |C|
    % where C is the smallest enclosing box

    iou = calculateIoU(pred, target);

    % Calculate enclosing box area
    x1_pred = pred(:, :, :, 1) - pred(:, :, :, 3) / 2;
    y1_pred = pred(:, :, :, 2) - pred(:, :, :, 4) / 2;
    x2_pred = pred(:, :, :, 1) + pred(:, :, :, 3) / 2;
    y2_pred = pred(:, :, :, 2) + pred(:, :, :, 4) / 2;

    x1_target = target(:, :, :, 1) - target(:, :, :, 3) / 2;
    y1_target = target(:, :, :, 2) - target(:, :, :, 4) / 2;
    x2_target = target(:, :, :, 1) + target(:, :, :, 3) / 2;
    y2_target = target(:, :, :, 2) + target(:, :, :, 4) / 2;

    % Enclosing box
    x1_c = min(x1_pred, x1_target);
    y1_c = min(y1_pred, y1_target);
    x2_c = max(x2_pred, x2_target);
    y2_c = max(y2_pred, y2_target);

    area_c = (x2_c - x1_c) .* (y2_c - y1_c);
    area_union = pred(:, :, :, 3) .* pred(:, :, :, 4) + ...
                 target(:, :, :, 3) .* target(:, :, :, 4) - ...
                 calculateIntersectionArea(pred, target);

    giou = iou - (area_c - area_union) ./ area_c;
    loss = mean(1 - giou, 'all');
end

function loss = diouLoss(pred, target)
    % DIoU Loss: Distance IoU
    % L = 1 - IoU + d²/c²
    % where d is the distance between box centers
    % and c is the diagonal length of the smallest enclosing box

    iou = calculateIoU(pred, target);

    % Center distances
    cx_pred = pred(:, :, :, 1);
    cy_pred = pred(:, :, :, 2);
    cx_target = target(:, :, :, 1);
    cy_target = target(:, :, :, 2);

    center_dist_sq = (cx_pred - cx_target).^2 + (cy_pred - cy_target).^2;

    % Enclosing box diagonal
    x1_pred = pred(:, :, :, 1) - pred(:, :, :, 3) / 2;
    y1_pred = pred(:, :, :, 2) - pred(:, :, :, 4) / 2;
    x2_pred = pred(:, :, :, 1) + pred(:, :, :, 3) / 2;
    y2_pred = pred(:, :, :, 2) + pred(:, :, :, 4) / 2;

    x1_target = target(:, :, :, 1) - target(:, :, :, 3) / 2;
    y1_target = target(:, :, :, 2) - target(:, :, :, 4) / 2;
    x2_target = target(:, :, :, 1) + target(:, :, :, 3) / 2;
    y2_target = target(:, :, :, 2) + target(:, :, :, 4) / 2;

    x1_c = min(x1_pred, x1_target);
    y1_c = min(y1_pred, y1_target);
    x2_c = max(x2_pred, x2_target);
    y2_c = max(y2_pred, y2_target);

    diagonal_sq = (x2_c - x1_c).^2 + (y2_c - y1_c).^2;

    diou = iou - center_dist_sq ./ (diagonal_sq + 1e-7);
    loss = mean(1 - diou, 'all');
end

function loss = ciouLoss(pred, target)
    % CIoU Loss: Complete IoU (RECOMMENDED FOR YOLOX)
    % L = 1 - IoU + d²/c² + αv
    % where v is the aspect ratio consistency
    % and α is the trade-off parameter

    iou = calculateIoU(pred, target);

    % DIoU component
    cx_pred = pred(:, :, :, 1);
    cy_pred = pred(:, :, :, 2);
    cx_target = target(:, :, :, 1);
    cy_target = target(:, :, :, 2);

    center_dist_sq = (cx_pred - cx_target).^2 + (cy_pred - cy_target).^2;

    x1_pred = pred(:, :, :, 1) - pred(:, :, :, 3) / 2;
    y1_pred = pred(:, :, :, 2) - pred(:, :, :, 4) / 2;
    x2_pred = pred(:, :, :, 1) + pred(:, :, :, 3) / 2;
    y2_pred = pred(:, :, :, 2) + pred(:, :, :, 4) / 2;

    x1_target = target(:, :, :, 1) - target(:, :, :, 3) / 2;
    y1_target = target(:, :, :, 2) - target(:, :, :, 4) / 2;
    x2_target = target(:, :, :, 1) + target(:, :, :, 3) / 2;
    y2_target = target(:, :, :, 2) + target(:, :, :, 4) / 2;

    x1_c = min(x1_pred, x1_target);
    y1_c = min(y1_pred, y1_target);
    x2_c = max(x2_pred, x2_target);
    y2_c = max(y2_pred, y2_target);

    diagonal_sq = (x2_c - x1_c).^2 + (y2_c - y1_c).^2;

    % Aspect ratio consistency
    w_pred = pred(:, :, :, 3);
    h_pred = pred(:, :, :, 4);
    w_target = target(:, :, :, 3);
    h_target = target(:, :, :, 4);

    v = (4 / pi^2) * (atan(w_target ./ (h_target + 1e-7)) - ...
                      atan(w_pred ./ (h_pred + 1e-7))).^2;

    % Trade-off parameter
    alpha = v ./ (1 - iou + v + 1e-7);

    ciou = iou - center_dist_sq ./ (diagonal_sq + 1e-7) - alpha .* v;
    loss = mean(1 - ciou, 'all');
end

%% Helper Functions for IoU

function iou = calculateIoU(pred, target)
    % Calculate Intersection over Union
    intersectionArea = calculateIntersectionArea(pred, target);
    unionArea = pred(:, :, :, 3) .* pred(:, :, :, 4) + ...
                target(:, :, :, 3) .* target(:, :, :, 4) - ...
                intersectionArea;
    iou = intersectionArea ./ (unionArea + 1e-7);
end

function area = calculateIntersectionArea(pred, target)
    % Calculate intersection area between predicted and target boxes

    % Convert center format to corner format
    x1_pred = pred(:, :, :, 1) - pred(:, :, :, 3) / 2;
    y1_pred = pred(:, :, :, 2) - pred(:, :, :, 4) / 2;
    x2_pred = pred(:, :, :, 1) + pred(:, :, :, 3) / 2;
    y2_pred = pred(:, :, :, 2) + pred(:, :, :, 4) / 2;

    x1_target = target(:, :, :, 1) - target(:, :, :, 3) / 2;
    y1_target = target(:, :, :, 2) - target(:, :, :, 4) / 2;
    x2_target = target(:, :, :, 1) + target(:, :, :, 3) / 2;
    y2_target = target(:, :, :, 2) + target(:, :, :, 4) / 2;

    % Intersection coordinates
    x1_inter = max(x1_pred, x1_target);
    y1_inter = max(y1_pred, y1_target);
    x2_inter = min(x2_pred, x2_target);
    y2_inter = min(y2_pred, y2_target);

    % Intersection area
    inter_w = max(x2_inter - x1_inter, 0);
    inter_h = max(y2_inter - y1_inter, 0);
    area = inter_w .* inter_h;
end

%% Classification Loss

function loss = classificationLoss(pred, target)
    % CLASSIFICATIONLOSS - Cross-entropy loss for classification
    %
    % Can use either:
    %   1. Standard Cross-Entropy
    %   2. Focal Loss (better for class imbalance)

    % Use cross-entropy
    loss = crossEntropyLoss(pred, target);

    % Alternative: Focal Loss (uncomment to use)
    % loss = focalLoss(pred, target, alpha=0.25, gamma=2.0);
end

function loss = crossEntropyLoss(pred, target)
    % Cross-Entropy Loss
    % L = -Σ target * log(pred)

    % Apply softmax to predictions
    pred = softmax(pred, 'DataFormat', 'SSCB');

    % Compute cross-entropy
    loss = -mean(target .* log(pred + 1e-7), 'all');
end

function loss = focalLoss(pred, target, alpha, gamma)
    % Focal Loss for addressing class imbalance
    % L = -α(1-p)^γ log(p)

    % Apply softmax
    pred = softmax(pred, 'DataFormat', 'SSCB');

    % Compute focal loss
    pt = target .* pred;
    focal_weight = alpha * (1 - pt).^gamma;
    loss = -mean(focal_weight .* log(pred + 1e-7), 'all');
end

%% Objectness Loss

function loss = objectnessLoss(pred, target)
    % OBJECTNESSLOSS - Binary cross-entropy for objectness score
    %
    % Inputs:
    %   pred - Predicted objectness scores [0, 1]
    %   target - Ground truth objectness {0, 1}
    %
    % Output:
    %   loss - Scalar loss value

    % Apply sigmoid to predictions
    pred = sigmoid(pred);

    % Binary cross-entropy
    loss = -mean(target .* log(pred + 1e-7) + ...
                 (1 - target) .* log(1 - pred + 1e-7), 'all');
end

%% Orientation Loss (CUSTOM FOR SAR)

function loss = orientationLoss(pred, target, delta)
    % ORIENTATIONLOSS - Huber loss for orientation estimation
    %
    % Inputs:
    %   pred - Predicted orientations (degrees or radians)
    %   target - Ground truth orientations
    %   delta - Huber loss threshold (default: 1.0)
    %
    % Output:
    %   loss - Scalar loss value
    %
    % Huber Loss combines L2 (MSE) for small errors and L1 (MAE) for large errors:
    %   L(e) = 0.5 * e²           if |e| ≤ δ
    %   L(e) = δ(|e| - 0.5δ)      if |e| > δ
    %
    % This makes it robust to outliers while maintaining smooth gradients

    if nargin < 3
        delta = 1.0;
    end

    % Calculate angular error (handle wrap-around)
    error = pred - target;

    % Wrap error to [-180, 180] for degree representation
    % Or [-π, π] for radian representation
    error = wrapAngle(error);

    % Compute Huber loss
    absError = abs(error);

    % Quadratic for small errors, linear for large errors
    isSmallError = absError <= delta;

    loss = zeros(size(error), 'like', error);
    loss(isSmallError) = 0.5 * error(isSmallError).^2;
    loss(~isSmallError) = delta * (absError(~isSmallError) - 0.5 * delta);

    loss = mean(loss, 'all');
end

function wrapped = wrapAngle(angle)
    % WRAPANGLE - Wrap angle to [-180, 180] degrees
    %
    % Handles angular wrap-around for orientation estimation
    % e.g., angle difference between 350° and 10° is 20°, not 340°

    wrapped = mod(angle + 180, 360) - 180;

    % For radians:
    % wrapped = mod(angle + pi, 2*pi) - pi;
end

function loss = circularLoss(pred, target)
    % CIRCULARLOSS - Alternative orientation loss using sine/cosine
    %
    % Instead of direct angle regression, predict sin(θ) and cos(θ)
    % This naturally handles periodicity

    % Convert angles to unit vectors
    pred_sin = sin(deg2rad(pred));
    pred_cos = cos(deg2rad(pred));

    target_sin = sin(deg2rad(target));
    target_cos = cos(deg2rad(target));

    % L2 loss on unit vectors
    loss = mean((pred_sin - target_sin).^2 + (pred_cos - target_cos).^2, 'all');
end

%% Additional Utility Functions

function y = sigmoid(x)
    % Sigmoid activation function
    y = 1 ./ (1 + exp(-x));
end

function y = softmax(x, varargin)
    % Softmax activation function
    % Already available in MATLAB, but defined here for completeness
    expX = exp(x - max(x, [], 3));  % Subtract max for numerical stability
    y = expX ./ sum(expX, 3);
end
