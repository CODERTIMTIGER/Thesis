function [trainData, valData, testData] = prepareData(dataPath, annotationPath, varargin)
%% PREPAREDATA - Prepare Mix_MSTAR dataset for YOLOX training
%
% Syntax:
%   [trainData, valData, testData] = prepareData(dataPath, annotationPath)
%   [trainData, valData, testData] = prepareData(__, Name, Value)
%
% Description:
%   Loads and preprocesses Mix_MSTAR SAR images and annotations for YOLOX
%   object detection. Handles both detection (bounding boxes) and
%   classification (vehicle types), with optional orientation labels.
%
% Inputs:
%   dataPath - Root directory containing train/val/test folders
%   annotationPath - Directory containing annotation files
%
% Name-Value Pairs:
%   'Split' - Data split ratios [train, val, test] (default: [0.7, 0.15, 0.15])
%   'IncludeOrientation' - Include orientation labels (default: true)
%   'AnnotationFormat' - 'yolo', 'coco', 'voc', or 'table' (default: 'table')
%   'Preprocess' - Apply SAR-specific preprocessing (default: true)
%
% Outputs:
%   trainData - Training dataset (table or datastore)
%   valData - Validation dataset
%   testData - Test dataset
%
% Example:
%   [trainData, valData, testData] = prepareData('data/', 'data/annotations/', ...
%       'IncludeOrientation', true, 'Split', [0.7, 0.15, 0.15]);
%
% See also: preprocessSARImage, createDatastore

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'dataPath', @ischar);
    addRequired(p, 'annotationPath', @ischar);
    addParameter(p, 'Split', [0.7, 0.15, 0.15], @(x) isnumeric(x) && sum(x)==1);
    addParameter(p, 'IncludeOrientation', true, @islogical);
    addParameter(p, 'AnnotationFormat', 'table', @ischar);
    addParameter(p, 'Preprocess', true, @islogical);
    addParameter(p, 'Shuffle', true, @islogical);
    parse(p, dataPath, annotationPath, varargin{:});

    opts = p.Results;

    fprintf('=== Preparing Mix_MSTAR Dataset ===\n');

    %% Load images and annotations
    fprintf('Loading images from: %s\n', dataPath);

    % Get all image files
    imgFiles = dir(fullfile(dataPath, '**', '*.jpg'));
    if isempty(imgFiles)
        imgFiles = [dir(fullfile(dataPath, '**', '*.png')); ...
                   dir(fullfile(dataPath, '**', '*.tif'))];
    end

    numImages = length(imgFiles);
    fprintf('Found %d images\n', numImages);

    if numImages == 0
        error('No images found in %s', dataPath);
    end

    %% Load annotations
    fprintf('Loading annotations from: %s\n', annotationPath);

    % Initialize data containers
    imagePaths = cell(numImages, 1);
    bboxes = cell(numImages, 1);
    labels = cell(numImages, 1);
    orientations = cell(numImages, 1);

    for i = 1:numImages
        imgPath = fullfile(imgFiles(i).folder, imgFiles(i).name);
        imagePaths{i} = imgPath;

        % Load corresponding annotation
        [~, imgName, ~] = fileparts(imgFiles(i).name);

        switch opts.AnnotationFormat
            case 'yolo'
                % YOLO format: class x_center y_center width height [orientation]
                annFile = fullfile(annotationPath, [imgName, '.txt']);
                [bboxes{i}, labels{i}, orientations{i}] = loadYOLOAnnotation(annFile, opts.IncludeOrientation);

            case 'coco'
                % COCO JSON format
                annFile = fullfile(annotationPath, 'annotations.json');
                [bboxes{i}, labels{i}, orientations{i}] = loadCOCOAnnotation(annFile, imgName, opts.IncludeOrientation);

            case 'voc'
                % Pascal VOC XML format
                annFile = fullfile(annotationPath, [imgName, '.xml']);
                [bboxes{i}, labels{i}, orientations{i}] = loadVOCAnnotation(annFile, opts.IncludeOrientation);

            case 'table'
                % Custom table format: ImagePath | BBox | Label | Orientation
                % Assume annotations are in a MAT file
                annFile = fullfile(annotationPath, 'annotations.mat');
                if i == 1
                    load(annFile, 'annotations');
                end
                idx = strcmp(annotations.ImagePath, imgPath);
                bboxes{i} = annotations.BBox{idx};
                labels{i} = annotations.Label{idx};
                if opts.IncludeOrientation
                    orientations{i} = annotations.Orientation{idx};
                else
                    orientations{i} = [];
                end
        end

        % Validate bounding boxes
        if ~isempty(bboxes{i})
            bboxes{i} = validateBoundingBoxes(bboxes{i});
        end
    end

    fprintf('Loaded annotations for %d images\n', numImages);

    %% Create dataset table
    if opts.IncludeOrientation
        dataTable = table(imagePaths, bboxes, labels, orientations, ...
            'VariableNames', {'ImagePath', 'BBox', 'Label', 'Orientation'});
    else
        dataTable = table(imagePaths, bboxes, labels, ...
            'VariableNames', {'ImagePath', 'BBox', 'Label'});
    end

    % Remove images with no annotations
    hasAnnotations = ~cellfun(@isempty, bboxes);
    dataTable = dataTable(hasAnnotations, :);
    fprintf('Removed %d images without annotations\n', sum(~hasAnnotations));
    fprintf('Final dataset size: %d images\n', height(dataTable));

    %% Shuffle dataset
    if opts.Shuffle
        rng(42); % For reproducibility
        shuffleIdx = randperm(height(dataTable));
        dataTable = dataTable(shuffleIdx, :);
    end

    %% Split dataset
    numTotal = height(dataTable);
    numTrain = floor(opts.Split(1) * numTotal);
    numVal = floor(opts.Split(2) * numTotal);
    numTest = numTotal - numTrain - numVal;

    trainData = dataTable(1:numTrain, :);
    valData = dataTable(numTrain+1:numTrain+numVal, :);
    testData = dataTable(numTrain+numVal+1:end, :);

    fprintf('\nDataset split:\n');
    fprintf('  Training:   %d images (%.1f%%)\n', height(trainData), 100*opts.Split(1));
    fprintf('  Validation: %d images (%.1f%%)\n', height(valData), 100*opts.Split(2));
    fprintf('  Testing:    %d images (%.1f%%)\n', height(testData), 100*opts.Split(3));

    %% Display dataset statistics
    displayDatasetStats(trainData, 'Training');
    displayDatasetStats(valData, 'Validation');
    displayDatasetStats(testData, 'Testing');

    fprintf('=== Data Preparation Complete ===\n\n');
end

%% Helper Functions

function [bbox, label, orientation] = loadYOLOAnnotation(annFile, includeOrientation)
    % Load YOLO format annotation: class x_center y_center width height [angle]
    if ~exist(annFile, 'file')
        bbox = [];
        label = {};
        orientation = [];
        return;
    end

    data = readmatrix(annFile);
    if isempty(data)
        bbox = [];
        label = {};
        orientation = [];
        return;
    end

    % Load class names
    cfg = config();
    classNames = cfg.CLASS_NAMES;

    % Convert YOLO format (normalized) to [x, y, width, height]
    % Note: YOLO uses center coordinates, need to convert to top-left
    numObjects = size(data, 1);
    bbox = zeros(numObjects, 4);
    label = cell(numObjects, 1);
    orientation = zeros(numObjects, 1);

    % Assume image size is known or read from image
    % For now, assume it's stored in the annotation or use default
    imgSize = 640; % Default size

    for i = 1:numObjects
        classIdx = data(i, 1) + 1; % YOLO classes are 0-indexed
        xCenter = data(i, 2) * imgSize;
        yCenter = data(i, 3) * imgSize;
        w = data(i, 4) * imgSize;
        h = data(i, 5) * imgSize;

        % Convert to [x, y, width, height] format (top-left corner)
        bbox(i, :) = [xCenter - w/2, yCenter - h/2, w, h];
        label{i} = classNames{classIdx};

        if includeOrientation && size(data, 2) >= 6
            orientation(i) = data(i, 6); % Orientation in degrees
        end
    end

    if ~includeOrientation
        orientation = [];
    end
end

function [bbox, label, orientation] = loadCOCOAnnotation(jsonFile, imgName, includeOrientation)
    % Load COCO format annotation (JSON)
    % This is a placeholder - implement based on your COCO structure
    bbox = [];
    label = {};
    orientation = [];

    warning('COCO format loading not fully implemented. Use custom loader.');
end

function [bbox, label, orientation] = loadVOCAnnotation(xmlFile, includeOrientation)
    % Load Pascal VOC format annotation (XML)
    % This is a placeholder - implement based on your VOC structure
    bbox = [];
    label = {};
    orientation = [];

    warning('VOC format loading not fully implemented. Use custom loader.');
end

function bbox = validateBoundingBoxes(bbox)
    % Ensure bounding boxes are valid (positive width/height, within image)
    if isempty(bbox)
        return;
    end

    % Ensure positive width and height
    bbox(:, 3) = max(bbox(:, 3), 1);
    bbox(:, 4) = max(bbox(:, 4), 1);

    % Ensure coordinates are positive
    bbox(:, 1:2) = max(bbox(:, 1:2), 1);
end

function displayDatasetStats(dataTable, splitName)
    % Display statistics about the dataset
    fprintf('\n%s Dataset Statistics:\n', splitName);
    fprintf('  Total images: %d\n', height(dataTable));

    % Count total objects
    totalObjects = sum(cellfun(@(x) size(x, 1), dataTable.BBox));
    fprintf('  Total objects: %d\n', totalObjects);
    fprintf('  Avg objects per image: %.2f\n', totalObjects / height(dataTable));

    % Class distribution
    allLabels = [];
    for i = 1:height(dataTable)
        allLabels = [allLabels; dataTable.Label{i}];
    end

    uniqueClasses = unique(allLabels);
    fprintf('  Number of classes: %d\n', length(uniqueClasses));

    % Display per-class counts
    fprintf('  Class distribution:\n');
    for i = 1:length(uniqueClasses)
        count = sum(strcmp(allLabels, uniqueClasses{i}));
        fprintf('    %s: %d (%.1f%%)\n', uniqueClasses{i}, count, 100*count/totalObjects);
    end

    % Bounding box statistics
    allBBoxes = vertcat(dataTable.BBox{:});
    if ~isempty(allBBoxes)
        avgWidth = mean(allBBoxes(:, 3));
        avgHeight = mean(allBBoxes(:, 4));
        avgArea = mean(allBBoxes(:, 3) .* allBBoxes(:, 4));
        fprintf('  Avg bbox size: %.1f x %.1f (area: %.1f)\n', avgWidth, avgHeight, avgArea);
    end
end
