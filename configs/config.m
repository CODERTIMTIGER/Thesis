%% YOLOX SAR Target Detection Configuration File
% Configuration for Mix_MSTAR dataset with YOLOX detection and classification
% Author: SAR Detection System
% Date: 2025-11-05

classdef config
    properties (Constant)
        %% Dataset Configuration
        % Mix_MSTAR Dataset paths
        DATA_ROOT = 'data/';
        TRAIN_IMG_PATH = 'data/train/';
        VAL_IMG_PATH = 'data/val/';
        TEST_IMG_PATH = 'data/test/';
        ANNOTATION_PATH = 'data/annotations/';

        % Image properties
        INPUT_SIZE = [640, 640, 3];  % YOLOX input size [H, W, C]
        IMAGE_SIZE = 640;             % Square image size

        %% MSTAR Vehicle Classes (10 standard classes)
        % Modify this list based on your specific Mix_MSTAR dataset
        CLASS_NAMES = {'BMP2', 'BTR70', 'T72', 'BTR60', 'BMP1', ...
                      '2S1', 'BRDM2', 'D7', 'T62', 'ZIL131'};
        NUM_CLASSES = 10;

        %% YOLOX Model Configuration
        % Available models: 'nano', 'tiny', 'small', 'medium', 'large', 'xlarge'
        YOLOX_MODEL = 'small';        % Start with small model
        PRETRAINED_MODEL = 'small-coco';  % Pretrained on COCO dataset

        % Model architecture
        BACKBONE = 'cspdarknet';      % CSPDarknet backbone
        NECK = 'panet';               % PANet for feature fusion
        HEAD_TYPE = 'decoupled';      % Decoupled head (classification + regression)

        % Detection scales
        DETECTION_SCALES = [8, 16, 32];  % Strides for P3, P4, P5
        FEATURE_MAP_SIZES = [80, 40, 20]; % For 640x640 input

        %% Orientation Detection
        DETECT_ORIENTATION = true;    % Enable orientation estimation
        ORIENTATION_BINS = 36;        % Discretize into 36 bins (10° each)
        % OR use continuous angle prediction
        ORIENTATION_MODE = 'continuous';  % 'continuous' or 'classification'
        ANGLE_RANGE = [-180, 180];    % Angle range in degrees

        %% Training Hyperparameters
        % Optimization
        OPTIMIZER = 'adam';
        INITIAL_LEARNING_RATE = 1e-3;
        LEARN_RATE_SCHEDULE = 'piecewise';
        LEARN_RATE_DROP_FACTOR = 0.1;
        LEARN_RATE_DROP_PERIOD = 30;
        L2_REGULARIZATION = 1e-4;
        GRADIENT_THRESHOLD = 10;

        % Training parameters
        MAX_EPOCHS = 100;
        MINI_BATCH_SIZE = 4;          % Adjust based on GPU memory
        VALIDATION_FREQUENCY = 50;    % Validate every N iterations
        CHECKPOINT_FREQUENCY = 10;    % Save checkpoint every N epochs

        % Early stopping
        PATIENCE = 10;                % Stop if no improvement for N epochs

        %% Loss Function Weights
        % Multi-task loss: L_total = λ_box*L_box + λ_obj*L_obj + λ_cls*L_cls + λ_ori*L_ori
        LAMBDA_BOX = 5.0;             % Bounding box loss weight
        LAMBDA_OBJ = 1.0;             % Objectness loss weight
        LAMBDA_CLS = 0.5;             % Classification loss weight
        LAMBDA_ORI = 0.3;             % Orientation loss weight

        % Huber loss parameters for orientation
        HUBER_DELTA = 1.0;            % Threshold for Huber loss

        % IoU loss type: 'iou', 'giou', 'diou', 'ciou'
        IOU_LOSS_TYPE = 'ciou';       % Complete IoU for better bbox regression

        %% Data Augmentation
        AUGMENTATION_ENABLED = true;

        % Augmentation parameters (SAR-specific)
        RANDOM_ROTATION = [-10, 10];  % Small rotations (degrees)
        RANDOM_SCALE = [0.8, 1.2];    % Scale factor range
        RANDOM_TRANSLATION = [-20, 20]; % Pixel translation
        HORIZONTAL_FLIP = true;
        VERTICAL_FLIP = false;        % Less common for SAR

        % Color/intensity augmentation
        BRIGHTNESS_JITTER = 0.2;      % For SAR intensity variation
        CONTRAST_JITTER = 0.2;

        % Advanced augmentations
        USE_MOSAIC = true;            % Mosaic augmentation (YOLOX default)
        USE_MIXUP = true;             % MixUp augmentation
        MOSAIC_PROB = 0.5;
        MIXUP_PROB = 0.15;

        %% SAR-Specific Preprocessing
        % SAR images often need special preprocessing
        APPLY_LOG_TRANSFORM = true;   % Log transformation for dynamic range
        NORMALIZE_INTENSITY = true;   % Normalize to [0, 1]
        SPECKLE_FILTER = 'none';      % 'none', 'lee', 'frost', 'kuan'

        % Convert single-channel SAR to 3-channel
        CONVERT_TO_RGB = true;        % Replicate channel for pretrained models

        %% Detection Parameters
        CONFIDENCE_THRESHOLD = 0.25;  % Minimum confidence for detection
        NMS_THRESHOLD = 0.45;         % Non-maximum suppression threshold
        MAX_DETECTIONS = 100;         % Maximum detections per image

        %% Evaluation Metrics
        IOU_THRESHOLDS = 0.5:0.05:0.95;  % For mAP calculation
        PRIMARY_METRIC = 'mAP_50';       % Primary metric for model selection

        % Orientation accuracy thresholds (degrees)
        ORI_ACCURACY_THRESHOLDS = [15, 30, 45];

        %% Hardware Configuration
        EXECUTION_ENVIRONMENT = 'gpu'; % 'auto', 'gpu', 'cpu', 'multi-gpu'
        GPU_ID = 1;                    % GPU device ID
        WORKERS = 4;                   % Number of parallel workers for data loading

        %% Model Save/Load
        MODEL_SAVE_PATH = 'models/checkpoints/';
        PRETRAINED_PATH = 'models/pretrained/';
        RESULTS_PATH = 'results/';

        % Model naming
        MODEL_NAME = 'yolox_sar_mstar';

        %% Visualization
        VISUALIZE_TRAINING = true;
        PLOT_FREQUENCY = 10;          % Plot every N epochs
        SAVE_VISUALIZATIONS = true;
        VIS_PATH = 'results/visualizations/';

        %% Debugging
        VERBOSE = true;
        DEBUG_MODE = false;           % Enable detailed logging
    end

    methods (Static)
        function printConfig()
            % Display current configuration
            fprintf('\n=== YOLOX SAR Detection Configuration ===\n');
            fprintf('Model: YOLOX-%s\n', config.YOLOX_MODEL);
            fprintf('Input Size: %dx%d\n', config.IMAGE_SIZE, config.IMAGE_SIZE);
            fprintf('Number of Classes: %d\n', config.NUM_CLASSES);
            fprintf('Orientation Detection: %s\n', string(config.DETECT_ORIENTATION));
            fprintf('Max Epochs: %d\n', config.MAX_EPOCHS);
            fprintf('Batch Size: %d\n', config.MINI_BATCH_SIZE);
            fprintf('Learning Rate: %.4f\n', config.INITIAL_LEARNING_RATE);
            fprintf('Execution Environment: %s\n', config.EXECUTION_ENVIRONMENT);
            fprintf('======================================\n\n');
        end

        function cfg = getTrainingOptions()
            % Get training options for trainingOptions function
            cfg = trainingOptions(config.OPTIMIZER, ...
                'MaxEpochs', config.MAX_EPOCHS, ...
                'MiniBatchSize', config.MINI_BATCH_SIZE, ...
                'InitialLearnRate', config.INITIAL_LEARNING_RATE, ...
                'LearnRateSchedule', config.LEARN_RATE_SCHEDULE, ...
                'LearnRateDropFactor', config.LEARN_RATE_DROP_FACTOR, ...
                'LearnRateDropPeriod', config.LEARN_RATE_DROP_PERIOD, ...
                'L2Regularization', config.L2_REGULARIZATION, ...
                'GradientThreshold', config.GRADIENT_THRESHOLD, ...
                'Shuffle', 'every-epoch', ...
                'ValidationFrequency', config.VALIDATION_FREQUENCY, ...
                'Verbose', config.VERBOSE, ...
                'Plots', 'training-progress', ...
                'ExecutionEnvironment', config.EXECUTION_ENVIRONMENT);
        end
    end
end
