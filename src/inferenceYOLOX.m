function [bboxes, scores, labels, orientations] = inferenceYOLOX(detector, image, varargin)
%% INFERENCEYOLOX - Run inference on SAR images using trained YOLOX
%
% Syntax:
%   [bboxes, scores, labels] = inferenceYOLOX(detector, image)
%   [bboxes, scores, labels, orientations] = inferenceYOLOX(detector, image, Name, Value)
%
% Description:
%   Performs object detection on SAR images using trained YOLOX detector.
%   Supports both standard yoloxObjectDetector and custom dlnetwork with
%   orientation estimation.
%
% Inputs:
%   detector - Trained YOLOX detector
%   image - Input SAR image (filepath or image array)
%
% Name-Value Pairs:
%   'ConfidenceThreshold' - Minimum detection confidence (default: 0.25)
%   'NMSThreshold' - Non-maximum suppression threshold (default: 0.45)
%   'Visualize' - Display detection results (default: false)
%   'SaveResults' - Save results to file (default: false)
%
% Outputs:
%   bboxes - Detected bounding boxes [x, y, width, height]
%   scores - Confidence scores for each detection
%   labels - Class labels for each detection
%   orientations - Orientation angles (if detector supports it)
%
% Example:
%   [bboxes, scores, labels, ori] = inferenceYOLOX(detector, 'test_image.jpg', ...
%       'ConfidenceThreshold', 0.3, 'Visualize', true);
%
% See also: trainYOLOX, evaluateModel, visualizeDetections

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'detector');
    addRequired(p, 'image');
    addParameter(p, 'ConfidenceThreshold', 0.25, @isnumeric);
    addParameter(p, 'NMSThreshold', 0.45, @isnumeric);
    addParameter(p, 'Visualize', false, @islogical);
    addParameter(p, 'SaveResults', false, @islogical);
    addParameter(p, 'OutputPath', 'results/detections/', @ischar);
    parse(p, detector, image, varargin{:});

    opts = p.Results;

    % Load configuration
    cfg = config();

    %% Load image
    if ischar(image) || isstring(image)
        % Image is filepath
        imgPath = image;
        img = imread(imgPath);
        [~, imgName, ~] = fileparts(imgPath);
    else
        % Image is array
        img = image;
        imgName = 'detection';
    end

    fprintf('Processing image: %s\n', imgName);
    fprintf('Image size: %d x %d x %d\n', size(img, 1), size(img, 2), size(img, 3));

    %% Preprocess image
    imgOriginal = img;  % Keep original for visualization
    img = preprocessImage(img, cfg);

    %% Run detection
    fprintf('Running detection...\n');
    tic;

    if isa(detector, 'yoloxObjectDetector')
        % Standard YOLOX detector
        [bboxes, scores, labels] = detect(detector, img, ...
            'Threshold', opts.ConfidenceThreshold, ...
            'SelectStrongest', true);
        orientations = [];  % Standard detector doesn't predict orientation

    elseif isa(detector, 'dlnetwork')
        % Custom dlnetwork detector with orientation
        [bboxes, scores, labels, orientations] = detectWithCustomNet(detector, img, cfg, opts);

    else
        error('Unknown detector type: %s', class(detector));
    end

    detectionTime = toc;
    fprintf('Detection complete in %.3f seconds\n', detectionTime);
    fprintf('Detected %d objects\n', size(bboxes, 1));

    %% Apply Non-Maximum Suppression (if not already done)
    if size(bboxes, 1) > 0
        [bboxes, scores, labels, orientations] = applyNMS(bboxes, scores, labels, ...
                                                           orientations, opts.NMSThreshold);
        fprintf('After NMS: %d objects\n', size(bboxes, 1));
    end

    %% Display results
    if ~isempty(bboxes)
        fprintf('\nDetection Results:\n');
        fprintf('%-5s %-15s %-10s %-10s\n', 'ID', 'Class', 'Confidence', 'Orientation');
        fprintf('%s\n', repmat('-', 1, 45));

        for i = 1:size(bboxes, 1)
            oriStr = '';
            if ~isempty(orientations)
                oriStr = sprintf('%.1f°', orientations(i));
            else
                oriStr = 'N/A';
            end

            fprintf('%-5d %-15s %-10.2f%% %-10s\n', ...
                    i, labels{i}, scores(i)*100, oriStr);
        end
    else
        fprintf('\nNo objects detected.\n');
    end

    %% Visualize
    if opts.Visualize
        visualizeDetections(imgOriginal, bboxes, scores, labels, orientations, ...
                           'Title', sprintf('Detections: %s', imgName));
    end

    %% Save results
    if opts.SaveResults
        saveDetectionResults(imgOriginal, bboxes, scores, labels, orientations, ...
                            imgName, opts.OutputPath, cfg);
    end
end

%% Custom Network Detection
function [bboxes, scores, labels, orientations] = detectWithCustomNet(net, img, cfg, opts)
    % Run detection using custom dlnetwork with orientation

    % Convert to dlarray
    X = dlarray(single(img), 'SSCB');

    % Move to GPU if available
    if canUseGPU()
        X = gpuArray(X);
    end

    % Forward pass
    predictions = predict(net, X);

    % Move back to CPU
    if canUseGPU()
        predictions = gather(predictions);
    end

    % Parse predictions
    % Predictions format: [1, H, W, features]
    % features = [x, y, w, h, objectness, class1, ..., classN, orientation]

    imgSize = size(img, 1);  % Assume square input
    numClasses = cfg.NUM_CLASSES;

    % Extract components
    predXY = predictions(1, :, :, 1:2);      % Center coordinates
    predWH = predictions(1, :, :, 3:4);      % Width and height
    predObj = predictions(1, :, :, 5);       % Objectness
    predCls = predictions(1, :, :, 6:5+numClasses);  % Class probabilities

    if cfg.DETECT_ORIENTATION
        predOri = predictions(1, :, :, 6+numClasses);  % Orientation
    end

    % Convert grid predictions to image coordinates
    [bboxes, scores, labels, orientations] = decodePredictions(predXY, predWH, predObj, ...
                                                                predCls, predOri, ...
                                                                imgSize, cfg, opts);
end

function [bboxes, scores, labels, orientations] = decodePredictions(predXY, predWH, predObj, ...
                                                                      predCls, predOri, ...
                                                                      imgSize, cfg, opts)
    % Decode YOLOX predictions to bounding boxes

    % Get grid dimensions
    [gridH, gridW] = size(predObj, [1, 2]);

    % Create grid coordinates
    [gridX, gridY] = meshgrid(0:gridW-1, 0:gridH-1);

    % Flatten predictions
    predXY = reshape(predXY, [], 2);
    predWH = reshape(predWH, [], 2);
    predObj = reshape(predObj, [], 1);
    predCls = reshape(predCls, [], size(predCls, 3));

    if ~isempty(predOri)
        predOri = reshape(predOri, [], 1);
    end

    gridX = reshape(gridX, [], 1);
    gridY = reshape(gridY, [], 1);

    % Calculate stride (image size / grid size)
    stride = imgSize / gridH;

    % Decode coordinates
    % YOLOX uses center-based prediction relative to grid cell
    centerX = (predXY(:, 1) + gridX) * stride;
    centerY = (predXY(:, 2) + gridY) * stride;
    width = exp(predWH(:, 1)) * stride;
    height = exp(predWH(:, 2)) * stride;

    % Convert to [x, y, w, h] format (top-left corner)
    bboxes = [centerX - width/2, centerY - height/2, width, height];

    % Apply sigmoid to objectness
    objScores = 1 ./ (1 + exp(-predObj));

    % Apply softmax to class predictions
    expCls = exp(predCls - max(predCls, [], 2));
    clsProbs = expCls ./ sum(expCls, 2);

    [clsScores, clsIdx] = max(clsProbs, [], 2);

    % Combined score = objectness * class_probability
    scores = objScores .* clsScores;

    % Filter by confidence threshold
    validIdx = scores >= opts.ConfidenceThreshold;

    bboxes = bboxes(validIdx, :);
    scores = scores(validIdx);
    clsIdx = clsIdx(validIdx);

    if ~isempty(predOri)
        orientations = predOri(validIdx);
    else
        orientations = [];
    end

    % Convert class indices to labels
    classNames = cfg.CLASS_NAMES;
    labels = classNames(clsIdx);

    % Clip bounding boxes to image boundaries
    bboxes = clipBBoxes(bboxes, imgSize, imgSize);
end

%% Non-Maximum Suppression
function [bboxes, scores, labels, orientations] = applyNMS(bboxes, scores, labels, ...
                                                            orientations, nmsThreshold)
    % Apply Non-Maximum Suppression to remove overlapping detections

    if isempty(bboxes)
        return;
    end

    % Convert labels to numeric if needed
    if iscell(labels)
        uniqueLabels = unique(labels);
        labelMap = containers.Map(uniqueLabels, 1:length(uniqueLabels));
        labelNums = cellfun(@(x) labelMap(x), labels);
    else
        labelNums = labels;
    end

    % Apply NMS per class
    selectedIdx = selectStrongestBboxMulticlass(bboxes, scores, labelNums, ...
                                                 'OverlapThreshold', nmsThreshold);

    % Filter detections
    bboxes = bboxes(selectedIdx, :);
    scores = scores(selectedIdx);
    if iscell(labels)
        labels = labels(selectedIdx);
    else
        labels = labels(selectedIdx);
    end

    if ~isempty(orientations)
        orientations = orientations(selectedIdx);
    end
end

%% Image Preprocessing
function img = preprocessImage(img, cfg)
    % Preprocess SAR image for inference

    % Convert to double
    img = im2double(img);

    % SAR-specific preprocessing
    if cfg.APPLY_LOG_TRANSFORM
        img = log(1 + img);
    end

    if cfg.NORMALIZE_INTENSITY
        imgMean = mean(img(:));
        imgStd = std(img(:));
        img = (img - imgMean) / (imgStd + eps);
    end

    % Convert to RGB if needed
    if cfg.CONVERT_TO_RGB && size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end

    % Resize to input size
    if ~isequal(size(img, 1:2), cfg.INPUT_SIZE(1:2))
        img = imresize(img, cfg.INPUT_SIZE(1:2));
    end
end

%% Utility Functions

function bboxes = clipBBoxes(bboxes, imgHeight, imgWidth)
    % Clip bounding boxes to image boundaries

    bboxes(:, 1) = max(1, min(bboxes(:, 1), imgWidth));
    bboxes(:, 2) = max(1, min(bboxes(:, 2), imgHeight));
    bboxes(:, 3) = max(1, min(bboxes(:, 3), imgWidth - bboxes(:, 1)));
    bboxes(:, 4) = max(1, min(bboxes(:, 4), imgHeight - bboxes(:, 2)));
end

function saveDetectionResults(img, bboxes, scores, labels, orientations, imgName, outputPath, cfg)
    % Save detection results

    % Create output directory if needed
    if ~exist(outputPath, 'dir')
        mkdir(outputPath);
    end

    % Save annotated image
    imgAnnotated = visualizeDetections(img, bboxes, scores, labels, orientations, ...
                                       'Display', false);

    imgPath = fullfile(outputPath, sprintf('%s_detected.png', imgName));
    imwrite(imgAnnotated, imgPath);

    % Save detection data
    detections = struct();
    detections.BBoxes = bboxes;
    detections.Scores = scores;
    detections.Labels = labels;
    detections.Orientations = orientations;

    dataPath = fullfile(outputPath, sprintf('%s_detections.mat', imgName));
    save(dataPath, 'detections');

    % Save as JSON for easy reading
    jsonPath = fullfile(outputPath, sprintf('%s_detections.json', imgName));
    jsonStr = jsonencode(detections);
    fid = fopen(jsonPath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);

    fprintf('Results saved to: %s\n', outputPath);
end
