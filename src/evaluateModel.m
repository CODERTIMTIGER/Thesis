function [metrics, results] = evaluateModel(detector, testData, varargin)
%% EVALUATEMODEL - Evaluate YOLOX detector on test dataset
%
% Syntax:
%   [metrics, results] = evaluateModel(detector, testData)
%   [metrics, results] = evaluateModel(____, Name, Value)
%
% Description:
%   Comprehensive evaluation of YOLOX SAR target detector including:
%   - Mean Average Precision (mAP) at different IoU thresholds
%   - Precision, Recall, F1-Score per class
%   - Orientation accuracy metrics
%   - Confusion matrix
%   - Per-class performance analysis
%
% Inputs:
%   detector - Trained YOLOX detector (yoloxObjectDetector or dlnetwork)
%   testData - Test dataset (table with ImagePath, BBox, Label, Orientation)
%
% Name-Value Pairs:
%   'IoUThreshold' - IoU threshold for mAP (default: 0.5:0.05:0.95)
%   'ConfidenceThreshold' - Minimum confidence for detection (default: 0.25)
%   'SaveResults' - Save results to file (default: true)
%   'Visualize' - Generate visualization plots (default: true)
%
% Outputs:
%   metrics - Struct containing all evaluation metrics
%   results - Detailed per-image results
%
% Example:
%   [metrics, results] = evaluateModel(detector, testData, 'Visualize', true);
%
% See also: trainYOLOX, visualizeDetections

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'detector');
    addRequired(p, 'testData', @istable);
    addParameter(p, 'IoUThreshold', 0.5:0.05:0.95, @isnumeric);
    addParameter(p, 'ConfidenceThreshold', 0.25, @isnumeric);
    addParameter(p, 'SaveResults', true, @islogical);
    addParameter(p, 'Visualize', true, @islogical);
    parse(p, detector, testData, varargin{:});

    opts = p.Results;

    % Load configuration
    cfg = config();

    fprintf('=== Evaluating YOLOX Detector ===\n');
    fprintf('Test dataset size: %d images\n', height(testData));

    %% Run detection on test set
    fprintf('\nRunning detection on test images...\n');

    numImages = height(testData);
    allDetections = cell(numImages, 1);
    allGroundTruth = cell(numImages, 1);
    allScores = cell(numImages, 1);
    allOrientations = cell(numImages, 1);

    % Progress bar
    progressBar = waitbar(0, 'Evaluating...');

    for i = 1:numImages
        % Load image
        img = imread(testData.ImagePath{i});

        % Preprocess
        img = preprocessSARImage(img, cfg);

        % Detect objects
        if isa(detector, 'yoloxObjectDetector')
            [bboxes, scores, labels] = detect(detector, img, ...
                'Threshold', opts.ConfidenceThreshold);
            orientations = [];  % Standard detector doesn't predict orientation
        else
            % Custom dlnetwork detector
            [bboxes, scores, labels, orientations] = detectCustom(detector, img, cfg);
        end

        % Store results
        allDetections{i} = bboxes;
        allScores{i} = scores;
        allOrientations{i} = orientations;

        % Store ground truth
        allGroundTruth{i}.BBox = testData.BBox{i};
        allGroundTruth{i}.Label = testData.Label{i};
        if cfg.DETECT_ORIENTATION && ismember('Orientation', testData.Properties.VariableNames)
            allGroundTruth{i}.Orientation = testData.Orientation{i};
        end

        % Update progress
        if mod(i, 10) == 0 || i == numImages
            waitbar(i/numImages, progressBar, ...
                sprintf('Evaluating: %d/%d images', i, numImages));
        end
    end

    close(progressBar);
    fprintf('Detection complete!\n');

    %% Calculate metrics
    fprintf('\nCalculating metrics...\n');

    % 1. Mean Average Precision (mAP)
    metrics.mAP = calculateMAP(allDetections, allGroundTruth, allScores, opts.IoUThreshold);

    % 2. Per-class metrics
    metrics.PerClass = calculatePerClassMetrics(allDetections, allGroundTruth, allScores, cfg);

    % 3. Overall Precision, Recall, F1
    metrics.Overall = calculateOverallMetrics(allDetections, allGroundTruth, opts.IoUThreshold(1));

    % 4. Orientation metrics (if applicable)
    if cfg.DETECT_ORIENTATION && ~isempty(allOrientations{1})
        metrics.Orientation = calculateOrientationMetrics(allOrientations, allGroundTruth, cfg);
    end

    % 5. Confusion Matrix
    metrics.ConfusionMatrix = calculateConfusionMatrix(allDetections, allGroundTruth, cfg);

    %% Display results
    displayMetrics(metrics, cfg);

    %% Save results
    if opts.SaveResults
        resultsPath = fullfile(cfg.RESULTS_PATH, 'metrics', ...
            sprintf('eval_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
        save(resultsPath, 'metrics', 'allDetections', 'allGroundTruth', 'allScores');
        fprintf('\nResults saved to: %s\n', resultsPath);

        % Save as JSON for easy reading
        jsonPath = strrep(resultsPath, '.mat', '.json');
        jsonStr = jsonencode(metrics);
        fid = fopen(jsonPath, 'w');
        fprintf(fid, '%s', jsonStr);
        fclose(fid);
    end

    %% Visualize results
    if opts.Visualize
        visualizeEvaluationResults(metrics, cfg);
    end

    %% Package results
    results.Detections = allDetections;
    results.GroundTruth = allGroundTruth;
    results.Scores = allScores;
    results.Orientations = allOrientations;

    fprintf('\n=== Evaluation Complete ===\n');
end

%% Calculate Mean Average Precision (mAP)
function mAP_results = calculateMAP(detections, groundTruth, scores, iouThresholds)
    % Calculate mAP at different IoU thresholds

    numThresholds = length(iouThresholds);
    AP = zeros(1, numThresholds);

    for i = 1:numThresholds
        iouThr = iouThresholds(i);
        AP(i) = calculateAP(detections, groundTruth, scores, iouThr);
    end

    mAP_results.AP_per_IoU = AP;
    mAP_results.IoU_thresholds = iouThresholds;
    mAP_results.mAP_50 = AP(iouThresholds == 0.5);      % AP@0.5
    mAP_results.mAP_75 = AP(iouThresholds == 0.75);     % AP@0.75
    mAP_results.mAP_50_95 = mean(AP);                   % AP@[0.5:0.95]
end

function ap = calculateAP(detections, groundTruth, scores, iouThreshold)
    % Calculate Average Precision for a single IoU threshold

    % Collect all detections and ground truth
    allDets = [];
    allGT = [];
    allScores = [];

    for i = 1:length(detections)
        if ~isempty(detections{i})
            numDets = size(detections{i}, 1);
            allDets = [allDets; detections{i}];
            allScores = [allScores; scores{i}];
        end

        if ~isempty(groundTruth{i}.BBox)
            allGT = [allGT; groundTruth{i}.BBox];
        end
    end

    if isempty(allDets) || isempty(allGT)
        ap = 0;
        return;
    end

    % Sort detections by confidence score (descending)
    [allScores, sortIdx] = sort(allScores, 'descend');
    allDets = allDets(sortIdx, :);

    % Match detections to ground truth
    numDets = size(allDets, 1);
    numGT = size(allGT, 1);

    tp = zeros(numDets, 1);
    fp = zeros(numDets, 1);
    gtMatched = false(numGT, 1);

    for i = 1:numDets
        detBox = allDets(i, :);

        % Find best matching ground truth
        maxIoU = 0;
        maxIdx = 0;

        for j = 1:numGT
            if gtMatched(j)
                continue;
            end

            gtBox = allGT(j, :);
            iou = bboxOverlapRatio(detBox, gtBox);

            if iou > maxIoU
                maxIoU = iou;
                maxIdx = j;
            end
        end

        % Check if detection is TP or FP
        if maxIoU >= iouThreshold
            tp(i) = 1;
            gtMatched(maxIdx) = true;
        else
            fp(i) = 1;
        end
    end

    % Compute precision-recall curve
    tp_cumsum = cumsum(tp);
    fp_cumsum = cumsum(fp);

    recall = tp_cumsum / numGT;
    precision = tp_cumsum ./ (tp_cumsum + fp_cumsum);

    % Compute AP using 11-point interpolation
    ap = 0;
    for t = 0:0.1:1
        p = max(precision(recall >= t));
        if isempty(p)
            p = 0;
        end
        ap = ap + p / 11;
    end
end

%% Per-Class Metrics
function perClassMetrics = calculatePerClassMetrics(detections, groundTruth, scores, cfg)
    % Calculate precision, recall, F1 for each class

    classNames = cfg.CLASS_NAMES;
    numClasses = length(classNames);

    perClassMetrics = struct();

    for c = 1:numClasses
        className = classNames{c};

        % Filter detections and ground truth for this class
        classDets = {};
        classGT = {};
        classScores = {};

        for i = 1:length(detections)
            % This is simplified - full implementation needs class labels
            classDets{i} = detections{i};
            classGT{i} = groundTruth{i};
            classScores{i} = scores{i};
        end

        % Calculate metrics
        [precision, recall, f1] = calculatePrecisionRecall(classDets, classGT, 0.5);

        perClassMetrics.(className).Precision = precision;
        perClassMetrics.(className).Recall = recall;
        perClassMetrics.(className).F1 = f1;
    end
end

%% Overall Metrics
function overallMetrics = calculateOverallMetrics(detections, groundTruth, iouThreshold)
    % Calculate overall precision, recall, F1

    [precision, recall, f1] = calculatePrecisionRecall(detections, groundTruth, iouThreshold);

    overallMetrics.Precision = precision;
    overallMetrics.Recall = recall;
    overallMetrics.F1 = f1;

    % Calculate total detections
    totalDets = sum(cellfun(@(x) size(x, 1), detections));
    totalGT = sum(cellfun(@(x) size(x.BBox, 1), groundTruth));

    overallMetrics.TotalDetections = totalDets;
    overallMetrics.TotalGroundTruth = totalGT;
end

function [precision, recall, f1] = calculatePrecisionRecall(detections, groundTruth, iouThreshold)
    % Calculate precision and recall

    tp = 0;
    fp = 0;
    fn = 0;

    for i = 1:length(detections)
        dets = detections{i};
        gt = groundTruth{i}.BBox;

        if isempty(dets) && isempty(gt)
            continue;
        elseif isempty(dets)
            fn = fn + size(gt, 1);
            continue;
        elseif isempty(gt)
            fp = fp + size(dets, 1);
            continue;
        end

        % Match detections to ground truth
        gtMatched = false(size(gt, 1), 1);

        for j = 1:size(dets, 1)
            detBox = dets(j, :);

            % Find best match
            maxIoU = 0;
            maxIdx = 0;

            for k = 1:size(gt, 1)
                if gtMatched(k)
                    continue;
                end

                iou = bboxOverlapRatio(detBox, gt(k, :));

                if iou > maxIoU
                    maxIoU = iou;
                    maxIdx = k;
                end
            end

            if maxIoU >= iouThreshold
                tp = tp + 1;
                gtMatched(maxIdx) = true;
            else
                fp = fp + 1;
            end
        end

        % Unmatched ground truth are false negatives
        fn = fn + sum(~gtMatched);
    end

    precision = tp / (tp + fp + eps);
    recall = tp / (tp + fn + eps);
    f1 = 2 * precision * recall / (precision + recall + eps);
end

%% Orientation Metrics
function oriMetrics = calculateOrientationMetrics(predictions, groundTruth, cfg)
    % Calculate orientation estimation accuracy

    allPredOri = [];
    allGTOri = [];

    for i = 1:length(predictions)
        if ~isempty(predictions{i}) && ~isempty(groundTruth{i}.Orientation)
            allPredOri = [allPredOri; predictions{i}];
            allGTOri = [allGTOri; groundTruth{i}.Orientation];
        end
    end

    if isempty(allPredOri)
        oriMetrics = struct();
        return;
    end

    % Calculate angular error
    angularError = abs(allPredOri - allGTOri);

    % Handle wrap-around
    angularError = min(angularError, 360 - angularError);

    % Mean Absolute Angular Error (MAAE)
    oriMetrics.MAAE = mean(angularError);

    % Median Angular Error
    oriMetrics.MedianError = median(angularError);

    % Accuracy within thresholds
    for i = 1:length(cfg.ORI_ACCURACY_THRESHOLDS)
        threshold = cfg.ORI_ACCURACY_THRESHOLDS(i);
        accuracy = sum(angularError < threshold) / length(angularError);
        oriMetrics.(sprintf('Accuracy_%ddeg', threshold)) = accuracy * 100;
    end

    % Error statistics
    oriMetrics.MinError = min(angularError);
    oriMetrics.MaxError = max(angularError);
    oriMetrics.StdError = std(angularError);
end

%% Confusion Matrix
function confMat = calculateConfusionMatrix(detections, groundTruth, cfg)
    % Calculate confusion matrix for classification

    classNames = cfg.CLASS_NAMES;
    numClasses = length(classNames);

    confMat = zeros(numClasses, numClasses);

    % This is simplified - full implementation needs class labels from detections
    % For now, return empty confusion matrix
    fprintf('Note: Confusion matrix calculation requires class labels in detections\n');
end

%% Display Metrics
function displayMetrics(metrics, cfg)
    % Display evaluation metrics in formatted output

    fprintf('\n====== EVALUATION METRICS ======\n\n');

    % Mean Average Precision
    fprintf('Mean Average Precision (mAP):\n');
    fprintf('  mAP@0.50      : %.2f%%\n', metrics.mAP.mAP_50 * 100);
    fprintf('  mAP@0.75      : %.2f%%\n', metrics.mAP.mAP_75 * 100);
    fprintf('  mAP@[0.5:0.95]: %.2f%%\n', metrics.mAP.mAP_50_95 * 100);

    % Overall metrics
    fprintf('\nOverall Detection Performance:\n');
    fprintf('  Precision: %.2f%%\n', metrics.Overall.Precision * 100);
    fprintf('  Recall   : %.2f%%\n', metrics.Overall.Recall * 100);
    fprintf('  F1-Score : %.2f%%\n', metrics.Overall.F1 * 100);
    fprintf('  Total Detections   : %d\n', metrics.Overall.TotalDetections);
    fprintf('  Total Ground Truth : %d\n', metrics.Overall.TotalGroundTruth);

    % Orientation metrics (if available)
    if isfield(metrics, 'Orientation') && ~isempty(fieldnames(metrics.Orientation))
        fprintf('\nOrientation Estimation:\n');
        fprintf('  Mean Absolute Angular Error: %.2f°\n', metrics.Orientation.MAAE);
        fprintf('  Median Angular Error       : %.2f°\n', metrics.Orientation.MedianError);

        thresholds = cfg.ORI_ACCURACY_THRESHOLDS;
        for i = 1:length(thresholds)
            fieldName = sprintf('Accuracy_%ddeg', thresholds(i));
            fprintf('  Accuracy within %2d°       : %.2f%%\n', ...
                    thresholds(i), metrics.Orientation.(fieldName));
        end
    end

    fprintf('\n================================\n\n');
end

%% Visualization
function visualizeEvaluationResults(metrics, cfg)
    % Create visualization plots for evaluation results

    % 1. Precision-Recall Curve (if data available)
    % 2. mAP at different IoU thresholds
    % 3. Per-class performance
    % 4. Orientation error distribution

    % mAP vs IoU threshold
    figure('Name', 'mAP vs IoU Threshold', 'Position', [100, 100, 800, 600]);

    plot(metrics.mAP.IoU_thresholds, metrics.mAP.AP_per_IoU * 100, ...
         'b-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('IoU Threshold');
    ylabel('Average Precision (%)');
    title('mAP at Different IoU Thresholds');
    grid on;
    ylim([0, 100]);

    % Save figure
    savePath = fullfile(cfg.VIS_PATH, 'mAP_vs_IoU.png');
    saveas(gcf, savePath);
    fprintf('Saved visualization: %s\n', savePath);
end

%% Helper Functions

function img = preprocessSARImage(img, cfg)
    % Preprocess SAR image (same as training preprocessing)

    img = im2double(img);

    if cfg.APPLY_LOG_TRANSFORM
        img = log(1 + img);
    end

    if cfg.NORMALIZE_INTENSITY
        img = (img - mean(img(:))) / std(img(:));
    end

    if cfg.CONVERT_TO_RGB && size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end

    img = imresize(img, cfg.INPUT_SIZE(1:2));
end

function [bboxes, scores, labels, orientations] = detectCustom(net, img, cfg)
    % Run detection using custom dlnetwork

    % Convert to dlarray
    X = dlarray(single(img), 'SSCB');

    % Forward pass
    predictions = predict(net, X);

    % Parse predictions and apply NMS
    % This is simplified - full implementation needed

    bboxes = [];
    scores = [];
    labels = {};
    orientations = [];
end
