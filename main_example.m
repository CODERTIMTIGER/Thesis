%% YOLOX SAR Target Detection - Main Example Script
%
% This script demonstrates the complete workflow for training and evaluating
% a YOLOX detector on Mix_MSTAR dataset with orientation estimation.
%
% Author: SAR Detection System
% Date: 2025-11-05

clear; clc; close all;

%% Setup
fprintf('=================================================\n');
fprintf('  YOLOX SAR Target Detection System\n');
fprintf('  Mix_MSTAR Dataset\n');
fprintf('=================================================\n\n');

% Add paths
addpath(genpath('src'));
addpath('configs');

% Load configuration
cfg = config();
cfg.printConfig();

%% Step 1: Data Preparation
fprintf('\n=== STEP 1: Data Preparation ===\n');

% Define paths
dataPath = 'data/';
annotationPath = 'data/annotations/';

% Check if data exists
if ~exist(dataPath, 'dir')
    error(['Data directory not found: ' dataPath '\n' ...
           'Please organize your Mix_MSTAR dataset according to README.md']);
end

% Prepare dataset
fprintf('Loading and preparing Mix_MSTAR dataset...\n');
[trainData, valData, testData] = prepareData(dataPath, annotationPath, ...
    'IncludeOrientation', cfg.DETECT_ORIENTATION, ...
    'Split', [0.7, 0.15, 0.15], ...
    'AnnotationFormat', 'table', ...
    'Shuffle', true);

fprintf('Data preparation complete!\n');
fprintf('  Training: %d images\n', height(trainData));
fprintf('  Validation: %d images\n', height(valData));
fprintf('  Testing: %d images\n', height(testData));

%% Step 2: Model Setup
fprintf('\n=== STEP 2: Model Setup ===\n');

% Choose training mode
trainingMode = 'custom';  % 'standard' or 'custom'

if strcmp(trainingMode, 'standard')
    % Standard YOLOX without orientation
    fprintf('Creating standard YOLOX detector...\n');
    detector = setupYOLOX('ModelSize', cfg.YOLOX_MODEL, ...
                         'NumClasses', cfg.NUM_CLASSES, ...
                         'Pretrained', true, ...
                         'PretrainedModel', cfg.PRETRAINED_MODEL, ...
                         'CustomHead', false, ...
                         'NetworkType', 'standard');
else
    % Custom YOLOX with orientation estimation
    fprintf('Creating custom YOLOX detector with orientation head...\n');
    detector = setupYOLOX('ModelSize', cfg.YOLOX_MODEL, ...
                         'NumClasses', cfg.NUM_CLASSES, ...
                         'Pretrained', true, ...
                         'CustomHead', true, ...
                         'NetworkType', 'custom');
end

fprintf('Model setup complete!\n');

%% Step 3: Training
fprintf('\n=== STEP 3: Training ===\n');

% Ask user if they want to train or load existing model
trainNewModel = input('Train new model? (1=Yes, 0=Load existing): ');

if trainNewModel
    fprintf('Starting training...\n');
    fprintf('This may take several hours depending on your hardware.\n\n');

    % Train model
    [trainedDetector, trainingInfo] = trainYOLOX(trainData, valData, ...
                                                 'Detector', detector, ...
                                                 'SaveCheckpoints', true, ...
                                                 'Verbose', true);

    % Save final model
    modelSavePath = fullfile(cfg.MODEL_SAVE_PATH, ...
        sprintf('%s_final_%s.mat', cfg.MODEL_NAME, datestr(now, 'yyyymmdd')));
    save(modelSavePath, 'trainedDetector', 'trainingInfo', 'cfg');
    fprintf('\nFinal model saved to: %s\n', modelSavePath);

    % Plot training history
    figure('Name', 'Training History');
    subplot(2,1,1);
    plot(trainingInfo.TrainingLoss, 'b-', 'LineWidth', 2);
    hold on;
    plot(trainingInfo.ValidationLoss, 'r-', 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Loss');
    title('Training and Validation Loss');
    legend('Training', 'Validation');
    grid on;

    subplot(2,1,2);
    plot(trainingInfo.ValidationLoss, 'r-', 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Validation Loss');
    title('Validation Loss (Zoomed)');
    grid on;

else
    % Load existing model
    [modelFile, modelPath] = uigetfile('models/checkpoints/*.mat', ...
                                       'Select trained model');
    if modelFile == 0
        error('No model selected. Exiting.');
    end

    fprintf('Loading model from: %s\n', fullfile(modelPath, modelFile));
    loaded = load(fullfile(modelPath, modelFile));
    trainedDetector = loaded.trainedDetector;
    if isfield(loaded, 'trainingInfo')
        trainingInfo = loaded.trainingInfo;
    end
    fprintf('Model loaded successfully!\n');
end

%% Step 4: Evaluation
fprintf('\n=== STEP 4: Evaluation ===\n');

% Evaluate on test set
fprintf('Evaluating model on test set...\n');
[metrics, results] = evaluateModel(trainedDetector, testData, ...
                                   'IoUThreshold', cfg.IOU_THRESHOLDS, ...
                                   'ConfidenceThreshold', cfg.CONFIDENCE_THRESHOLD, ...
                                   'SaveResults', true, ...
                                   'Visualize', true);

% Display comprehensive results
fprintf('\n========================================\n');
fprintf('         FINAL EVALUATION RESULTS\n');
fprintf('========================================\n\n');

fprintf('Detection Performance:\n');
fprintf('  mAP@0.50      : %.2f%%\n', metrics.mAP.mAP_50 * 100);
fprintf('  mAP@0.75      : %.2f%%\n', metrics.mAP.mAP_75 * 100);
fprintf('  mAP@[0.5:0.95]: %.2f%%\n', metrics.mAP.mAP_50_95 * 100);
fprintf('  Precision     : %.2f%%\n', metrics.Overall.Precision * 100);
fprintf('  Recall        : %.2f%%\n', metrics.Overall.Recall * 100);
fprintf('  F1-Score      : %.2f%%\n\n', metrics.Overall.F1 * 100);

if isfield(metrics, 'Orientation') && ~isempty(fieldnames(metrics.Orientation))
    fprintf('Orientation Estimation:\n');
    fprintf('  Mean Abs Angular Error: %.2f°\n', metrics.Orientation.MAAE);
    fprintf('  Median Angular Error  : %.2f°\n', metrics.Orientation.MedianError);

    for i = 1:length(cfg.ORI_ACCURACY_THRESHOLDS)
        threshold = cfg.ORI_ACCURACY_THRESHOLDS(i);
        fieldName = sprintf('Accuracy_%ddeg', threshold);
        if isfield(metrics.Orientation, fieldName)
            fprintf('  Accuracy within %2d°  : %.2f%%\n', ...
                    threshold, metrics.Orientation.(fieldName));
        end
    end
    fprintf('\n');
end

fprintf('========================================\n\n');

%% Step 5: Demo Inference
fprintf('\n=== STEP 5: Demo Inference ===\n');

% Run inference on sample test images
numSamples = min(5, height(testData));
fprintf('Running inference on %d sample images...\n', numSamples);

figure('Name', 'Sample Detections', 'Position', [50, 50, 1400, 900]);

for i = 1:numSamples
    % Load test image
    imgPath = testData.ImagePath{i};
    img = imread(imgPath);

    % Run inference
    [bboxes, scores, labels, orientations] = inferenceYOLOX(trainedDetector, img, ...
        'ConfidenceThreshold', cfg.CONFIDENCE_THRESHOLD, ...
        'NMSThreshold', cfg.NMS_THRESHOLD, ...
        'Visualize', false);

    % Visualize
    subplot(2, 3, i);
    imgAnnotated = visualizeDetections(img, bboxes, scores, labels, orientations, ...
                                      'Display', false, ...
                                      'ShowOrientation', cfg.DETECT_ORIENTATION, ...
                                      'ShowConfidence', true);
    imshow(imgAnnotated);
    [~, imgName, ~] = fileparts(imgPath);
    title(sprintf('%s (%d detections)', imgName, size(bboxes, 1)), ...
          'Interpreter', 'none', 'FontSize', 10);
    axis off;
end

sgtitle('Sample Detection Results', 'FontSize', 14, 'FontWeight', 'bold');

%% Step 6: Interactive Inference
fprintf('\n=== STEP 6: Interactive Inference ===\n');

continueInference = input('Run interactive inference? (1=Yes, 0=No): ');

while continueInference
    % Select image
    [imgFile, imgPath] = uigetfile({'*.jpg;*.png;*.tif', 'Image Files'}, ...
                                   'Select SAR image for inference');

    if imgFile == 0
        break;
    end

    fullImgPath = fullfile(imgPath, imgFile);
    fprintf('\nProcessing: %s\n', imgFile);

    % Run inference
    [bboxes, scores, labels, orientations] = inferenceYOLOX(trainedDetector, ...
        fullImgPath, ...
        'ConfidenceThreshold', cfg.CONFIDENCE_THRESHOLD, ...
        'Visualize', true, ...
        'SaveResults', true);

    % Display results
    fprintf('\nResults:\n');
    if isempty(bboxes)
        fprintf('  No detections found.\n');
    else
        for j = 1:size(bboxes, 1)
            if ~isempty(orientations)
                fprintf('  %d. %s (%.1f%%) - Orientation: %.1f°\n', ...
                        j, labels{j}, scores(j)*100, orientations(j));
            else
                fprintf('  %d. %s (%.1f%%)\n', j, labels{j}, scores(j)*100);
            end
        end
    end

    continueInference = input('\nProcess another image? (1=Yes, 0=No): ');
end

%% Summary
fprintf('\n=================================================\n');
fprintf('  Processing Complete!\n');
fprintf('=================================================\n\n');

fprintf('Summary:\n');
fprintf('  Model: YOLOX-%s\n', cfg.YOLOX_MODEL);
fprintf('  Dataset: Mix_MSTAR\n');
fprintf('  Classes: %d\n', cfg.NUM_CLASSES);
fprintf('  Orientation: %s\n', string(cfg.DETECT_ORIENTATION));
fprintf('  Best mAP@0.5: %.2f%%\n', metrics.mAP.mAP_50 * 100);

fprintf('\nResults saved in:\n');
fprintf('  Models: %s\n', cfg.MODEL_SAVE_PATH);
fprintf('  Metrics: %s\n', fullfile(cfg.RESULTS_PATH, 'metrics/'));
fprintf('  Visualizations: %s\n', cfg.VIS_PATH);

fprintf('\nThank you for using YOLOX SAR Detection System!\n\n');

%% Cleanup
% Optionally clear large variables
% clear trainData valData testData detector trainedDetector
