function [trainedDetector, trainingInfo] = trainYOLOX(trainData, valData, varargin)
%% TRAINYOLOX - Train YOLOX model for SAR target detection
%
% Syntax:
%   [trainedDetector, trainingInfo] = trainYOLOX(trainData, valData)
%   [trainedDetector, trainingInfo] = trainYOLOX(__, Name, Value)
%
% Description:
%   Trains a YOLOX object detector on Mix_MSTAR dataset with custom
%   orientation estimation. Supports both standard yoloxObjectDetector
%   and custom dlnetwork architectures.
%
% Inputs:
%   trainData - Training dataset (table with ImagePath, BBox, Label, Orientation)
%   valData - Validation dataset
%
% Name-Value Pairs:
%   'Detector' - Pre-initialized detector (default: create new)
%   'ResumeFrom' - Checkpoint path to resume training (default: '')
%   'SaveCheckpoints' - Save checkpoints during training (default: true)
%   'Verbose' - Display detailed training info (default: true)
%
% Outputs:
%   trainedDetector - Trained YOLOX detector
%   trainingInfo - Struct with training history and metrics
%
% Example:
%   [detector, info] = trainYOLOX(trainData, valData, 'SaveCheckpoints', true);
%
% See also: setupYOLOX, prepareData, evaluateModel

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'trainData', @istable);
    addRequired(p, 'valData', @istable);
    addParameter(p, 'Detector', [], @(x) isa(x, 'yoloxObjectDetector') || isa(x, 'dlnetwork'));
    addParameter(p, 'ResumeFrom', '', @ischar);
    addParameter(p, 'SaveCheckpoints', true, @islogical);
    addParameter(p, 'Verbose', true, @islogical);
    parse(p, trainData, valData, varargin{:});

    opts = p.Results;

    % Load configuration
    cfg = config();
    cfg.printConfig();

    %% Initialize detector
    fprintf('=== Initializing YOLOX Detector ===\n');

    if isempty(opts.Detector)
        % Create new detector
        detector = setupYOLOX('ModelSize', cfg.YOLOX_MODEL, ...
                             'NumClasses', cfg.NUM_CLASSES, ...
                             'Pretrained', true, ...
                             'CustomHead', cfg.DETECT_ORIENTATION);
    else
        % Use provided detector
        detector = opts.Detector;
    end

    % Resume from checkpoint if specified
    if ~isempty(opts.ResumeFrom)
        fprintf('Resuming training from checkpoint: %s\n', opts.ResumeFrom);
        detector = loadCheckpoint(opts.ResumeFrom);
    end

    %% Prepare data
    fprintf('\n=== Preparing Training Data ===\n');

    % Create datastores for efficient data loading
    trainDS = createImageDatastore(trainData, cfg);
    valDS = createImageDatastore(valData, cfg);

    fprintf('Training samples: %d\n', height(trainData));
    fprintf('Validation samples: %d\n', height(valData));

    %% Setup training
    fprintf('\n=== Setting Up Training ===\n');

    % Check detector type and choose training method
    if isa(detector, 'yoloxObjectDetector')
        % Standard YOLOX training using Computer Vision Toolbox
        [trainedDetector, trainingInfo] = trainStandardYOLOX(detector, trainDS, valDS, cfg, opts);
    elseif isa(detector, 'dlnetwork')
        % Custom training loop for dlnetwork with orientation
        [trainedDetector, trainingInfo] = trainCustomYOLOX(detector, trainDS, valDS, cfg, opts);
    else
        error('Unknown detector type: %s', class(detector));
    end

    %% Save final model
    if opts.SaveCheckpoints
        savePath = fullfile(cfg.MODEL_SAVE_PATH, sprintf('%s_final.mat', cfg.MODEL_NAME));
        fprintf('\nSaving final model to: %s\n', savePath);
        save(savePath, 'trainedDetector', 'trainingInfo', 'cfg');
    end

    fprintf('\n=== Training Complete ===\n');
end

%% Standard YOLOX Training (Computer Vision Toolbox)
function [detector, info] = trainStandardYOLOX(detector, trainDS, valDS, cfg, opts)
    % Train using built-in trainYOLOXObjectDetector function

    fprintf('Using standard YOLOX training...\n');

    % Create training options
    trainingOpts = trainingOptions(cfg.OPTIMIZER, ...
        'MaxEpochs', cfg.MAX_EPOCHS, ...
        'MiniBatchSize', cfg.MINI_BATCH_SIZE, ...
        'InitialLearnRate', cfg.INITIAL_LEARNING_RATE, ...
        'LearnRateSchedule', cfg.LEARN_RATE_SCHEDULE, ...
        'LearnRateDropFactor', cfg.LEARN_RATE_DROP_FACTOR, ...
        'LearnRateDropPeriod', cfg.LEARN_RATE_DROP_PERIOD, ...
        'L2Regularization', cfg.L2_REGULARIZATION, ...
        'GradientThreshold', cfg.GRADIENT_THRESHOLD, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', valDS, ...
        'ValidationFrequency', cfg.VALIDATION_FREQUENCY, ...
        'Verbose', opts.Verbose, ...
        'Plots', 'training-progress', ...
        'CheckpointPath', cfg.MODEL_SAVE_PATH, ...
        'ExecutionEnvironment', cfg.EXECUTION_ENVIRONMENT);

    % Train detector
    [detector, info] = trainYOLOXObjectDetector(trainDS, detector, trainingOpts);
end

%% Custom YOLOX Training (dlnetwork with Orientation)
function [net, info] = trainCustomYOLOX(net, trainDS, valDS, cfg, opts)
    % Custom training loop for dlnetwork with orientation estimation

    fprintf('Using custom YOLOX training with orientation...\n');

    % Initialize training state
    iteration = 0;
    epoch = 0;
    bestValLoss = inf;
    patienceCounter = 0;

    % Training history
    trainingLoss = [];
    validationLoss = [];
    trainingMetrics = struct();

    % Setup optimizer
    switch lower(cfg.OPTIMIZER)
        case 'adam'
            avgGrad = [];
            avgGradSq = [];
        case 'sgdm'
            velocity = [];
        otherwise
            error('Unknown optimizer: %s', cfg.OPTIMIZER);
    end

    learningRate = cfg.INITIAL_LEARNING_RATE;

    % Training loop
    fprintf('\nStarting training for %d epochs...\n', cfg.MAX_EPOCHS);
    fprintf('%-10s %-15s %-15s %-15s %-15s %-15s\n', ...
            'Epoch', 'Train Loss', 'Val Loss', 'BBox Loss', 'Cls Loss', 'Ori Loss');
    fprintf('%s\n', repmat('-', 1, 95));

    for epoch = 1:cfg.MAX_EPOCHS
        % Shuffle training data
        trainDS = shuffle(trainDS);

        % Reset epoch metrics
        epochLoss = 0;
        epochBBoxLoss = 0;
        epochClsLoss = 0;
        epochOriLoss = 0;
        numBatches = 0;

        % Mini-batch training
        while hasdata(trainDS)
            iteration = iteration + 1;
            numBatches = numBatches + 1;

            % Read mini-batch
            [X, targets] = read(trainDS);

            % Convert to dlarray
            X = dlarray(X, 'SSCB');  % Spatial, Spatial, Channel, Batch

            % Move to GPU if available
            if canUseGPU()
                X = gpuArray(X);
                targets = structfun(@gpuArray, targets, 'UniformOutput', false);
            end

            % Compute loss and gradients
            [loss, gradients, lossComponents] = dlfeval(@customLossFunctions.yoloxLoss, ...
                                                         net, X, targets, targets, cfg);

            % Update learnable parameters
            [net, avgGrad, avgGradSq] = adamUpdate(net, gradients, avgGrad, avgGradSq, ...
                                                    iteration, learningRate);

            % Accumulate losses
            epochLoss = epochLoss + extractdata(loss);
            epochBBoxLoss = epochBBoxLoss + lossComponents.BBox;
            epochClsLoss = epochClsLoss + lossComponents.Classification;
            if cfg.DETECT_ORIENTATION
                epochOriLoss = epochOriLoss + lossComponents.Orientation;
            end
        end

        % Average epoch losses
        epochLoss = epochLoss / numBatches;
        epochBBoxLoss = epochBBoxLoss / numBatches;
        epochClsLoss = epochClsLoss / numBatches;
        epochOriLoss = epochOriLoss / numBatches;

        % Validation
        valLoss = validateModel(net, valDS, cfg);

        % Store training history
        trainingLoss(end+1) = epochLoss;
        validationLoss(end+1) = valLoss;

        % Display progress
        fprintf('%-10d %-15.4f %-15.4f %-15.4f %-15.4f %-15.4f\n', ...
                epoch, epochLoss, valLoss, epochBBoxLoss, epochClsLoss, epochOriLoss);

        % Learning rate schedule
        if strcmpi(cfg.LEARN_RATE_SCHEDULE, 'piecewise')
            if mod(epoch, cfg.LEARN_RATE_DROP_PERIOD) == 0
                learningRate = learningRate * cfg.LEARN_RATE_DROP_FACTOR;
                fprintf('Learning rate decreased to: %.6f\n', learningRate);
            end
        end

        % Save checkpoint
        if opts.SaveCheckpoints && mod(epoch, cfg.CHECKPOINT_FREQUENCY) == 0
            checkpointPath = fullfile(cfg.MODEL_SAVE_PATH, ...
                sprintf('%s_epoch_%d.mat', cfg.MODEL_NAME, epoch));
            save(checkpointPath, 'net', 'epoch', 'trainingLoss', 'validationLoss');
            fprintf('Checkpoint saved: %s\n', checkpointPath);
        end

        % Early stopping
        if valLoss < bestValLoss
            bestValLoss = valLoss;
            patienceCounter = 0;

            % Save best model
            if opts.SaveCheckpoints
                bestModelPath = fullfile(cfg.MODEL_SAVE_PATH, ...
                    sprintf('%s_best.mat', cfg.MODEL_NAME));
                save(bestModelPath, 'net', 'epoch', 'bestValLoss');
            end
        else
            patienceCounter = patienceCounter + 1;
            if patienceCounter >= cfg.PATIENCE
                fprintf('\nEarly stopping triggered at epoch %d\n', epoch);
                break;
            end
        end

        % Plot training progress
        if cfg.VISUALIZE_TRAINING && mod(epoch, cfg.PLOT_FREQUENCY) == 0
            plotTrainingProgress(trainingLoss, validationLoss, epoch);
        end
    end

    % Return training info
    info.TrainingLoss = trainingLoss;
    info.ValidationLoss = validationLoss;
    info.BestValidationLoss = bestValLoss;
    info.FinalEpoch = epoch;
    info.Metrics = trainingMetrics;
end

%% Validation Function
function valLoss = validateModel(net, valDS, cfg)
    % Compute validation loss

    totalLoss = 0;
    numBatches = 0;

    reset(valDS);

    while hasdata(valDS)
        numBatches = numBatches + 1;

        % Read batch
        [X, targets] = read(valDS);

        % Convert to dlarray
        X = dlarray(X, 'SSCB');

        % Move to GPU if available
        if canUseGPU()
            X = gpuArray(X);
            targets = structfun(@gpuArray, targets, 'UniformOutput', false);
        end

        % Forward pass only (no gradients)
        [loss, ~, ~] = customLossFunctions.yoloxLoss(net, X, targets, targets, cfg);

        totalLoss = totalLoss + extractdata(loss);
    end

    valLoss = totalLoss / numBatches;
    reset(valDS);
end

%% Optimizer Update Functions

function [net, avgGrad, avgGradSq] = adamUpdate(net, gradients, avgGrad, avgGradSq, iteration, learningRate)
    % Adam optimizer update
    % Parameters
    beta1 = 0.9;    % Exponential decay rate for first moment
    beta2 = 0.999;  % Exponential decay rate for second moment
    epsilon = 1e-8;

    % Initialize moments if first iteration
    if isempty(avgGrad)
        avgGrad = gradients;
        avgGradSq = gradients .* gradients;
    else
        % Update biased first moment estimate
        avgGrad = beta1 .* avgGrad + (1 - beta1) .* gradients;

        % Update biased second raw moment estimate
        avgGradSq = beta2 .* avgGradSq + (1 - beta2) .* (gradients .* gradients);
    end

    % Bias correction
    avgGradCorrected = avgGrad ./ (1 - beta1^iteration);
    avgGradSqCorrected = avgGradSq ./ (1 - beta2^iteration);

    % Update parameters
    net.Learnables.Value = net.Learnables.Value - ...
        learningRate .* avgGradCorrected ./ (sqrt(avgGradSqCorrected) + epsilon);
end

%% Data Utility Functions

function ds = createImageDatastore(dataTable, cfg)
    % Create datastore for efficient data loading

    % This is a simplified version - full implementation would use
    % imageDatastore and boxLabelDatastore combined with transform

    % For now, create a custom datastore wrapper
    ds = CustomYOLOXDatastore(dataTable, cfg);
end

function detector = loadCheckpoint(checkpointPath)
    % Load detector from checkpoint
    fprintf('Loading checkpoint from: %s\n', checkpointPath);
    data = load(checkpointPath);
    detector = data.net;
end

%% Visualization
function plotTrainingProgress(trainLoss, valLoss, currentEpoch)
    % Plot training and validation loss

    figure(100);
    clf;

    epochs = 1:length(trainLoss);

    plot(epochs, trainLoss, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
    hold on;
    plot(epochs, valLoss, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Loss');
    hold off;

    xlabel('Epoch');
    ylabel('Loss');
    title(sprintf('Training Progress (Epoch %d)', currentEpoch));
    legend('Location', 'best');
    grid on;

    drawnow;
end

%% Custom Datastore Class
classdef CustomYOLOXDatastore < matlab.io.Datastore
    % Custom datastore for YOLOX training with augmentation

    properties
        DataTable
        Config
        CurrentIndex
        NumObservations
    end

    methods
        function ds = CustomYOLOXDatastore(dataTable, cfg)
            ds.DataTable = dataTable;
            ds.Config = cfg;
            ds.CurrentIndex = 1;
            ds.NumObservations = height(dataTable);
        end

        function tf = hasdata(ds)
            tf = ds.CurrentIndex <= ds.NumObservations;
        end

        function [data, info] = read(ds)
            % Read one observation
            row = ds.DataTable(ds.CurrentIndex, :);
            ds.CurrentIndex = ds.CurrentIndex + 1;

            % Load and preprocess image
            img = imread(row.ImagePath{1});
            img = preprocessImage(img, ds.Config);

            % Load targets
            targets.BBox = row.BBox{1};
            targets.Label = row.Label{1};
            if ds.Config.DETECT_ORIENTATION
                targets.Orientation = row.Orientation{1};
            end

            data = {img, targets};
            info = [];
        end

        function reset(ds)
            ds.CurrentIndex = 1;
        end

        function ds = shuffle(ds)
            % Shuffle data
            shuffleIdx = randperm(ds.NumObservations);
            ds.DataTable = ds.DataTable(shuffleIdx, :);
            reset(ds);
        end
    end
end

%% Image Preprocessing
function img = preprocessImage(img, cfg)
    % Preprocess SAR image for YOLOX

    % Convert to double
    img = im2double(img);

    % SAR-specific preprocessing
    if cfg.APPLY_LOG_TRANSFORM
        img = log(1 + img);
    end

    if cfg.NORMALIZE_INTENSITY
        img = (img - mean(img(:))) / std(img(:));
    end

    % Convert to RGB if needed
    if cfg.CONVERT_TO_RGB && size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end

    % Resize to input size
    img = imresize(img, cfg.INPUT_SIZE(1:2));

    % Data augmentation would go here
    % (rotation, translation, flipping, etc.)
end
