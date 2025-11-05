function outputImg = visualizeDetections(img, bboxes, scores, labels, orientations, varargin)
%% VISUALIZEDETECTIONS - Visualize YOLOX detection results on SAR images
%
% Syntax:
%   outputImg = visualizeDetections(img, bboxes, scores, labels, orientations)
%   outputImg = visualizeDetections(__, Name, Value)
%
% Description:
%   Creates annotated visualization of object detection results including
%   bounding boxes, class labels, confidence scores, and orientation arrows.
%
% Inputs:
%   img - Input image (grayscale or RGB)
%   bboxes - Bounding boxes [x, y, width, height] (Nx4 matrix)
%   scores - Confidence scores (Nx1 vector)
%   labels - Class labels (Nx1 cell array or categorical)
%   orientations - Orientation angles in degrees (Nx1 vector, optional)
%
% Name-Value Pairs:
%   'Display' - Display the image (default: true)
%   'Title' - Figure title (default: 'Detections')
%   'LineWidth' - Bounding box line width (default: 2)
%   'FontSize' - Label font size (default: 10)
%   'ShowOrientation' - Draw orientation arrows (default: true)
%   'ShowConfidence' - Show confidence scores (default: true)
%   'ColorMap' - Color scheme: 'default', 'class', or Nx3 RGB array
%
% Outputs:
%   outputImg - Annotated image
%
% Example:
%   img = visualizeDetections(img, bboxes, scores, labels, orientations, ...
%       'Title', 'SAR Target Detection', 'ShowOrientation', true);
%
% See also: inferenceYOLOX, evaluateModel

    %% Parse inputs
    p = inputParser;
    addRequired(p, 'img');
    addRequired(p, 'bboxes');
    addRequired(p, 'scores');
    addRequired(p, 'labels');
    addRequired(p, 'orientations');
    addParameter(p, 'Display', true, @islogical);
    addParameter(p, 'Title', 'Detections', @ischar);
    addParameter(p, 'LineWidth', 2, @isnumeric);
    addParameter(p, 'FontSize', 10, @isnumeric);
    addParameter(p, 'ShowOrientation', true, @islogical);
    addParameter(p, 'ShowConfidence', true, @islogical);
    addParameter(p, 'ColorMap', 'default', @(x) ischar(x) || isnumeric(x));
    parse(p, img, bboxes, scores, labels, orientations, varargin{:});

    opts = p.Results;

    %% Prepare image
    outputImg = img;

    % Convert grayscale to RGB for colored annotations
    if size(outputImg, 3) == 1
        outputImg = repmat(outputImg, [1, 1, 3]);
    end

    % Normalize to [0, 1] if needed
    if ~isfloat(outputImg)
        outputImg = im2double(outputImg);
    end

    %% Handle empty detections
    if isempty(bboxes) || size(bboxes, 1) == 0
        if opts.Display
            figure('Name', opts.Title);
            imshow(outputImg);
            title(opts.Title);
            text(10, 30, 'No detections', 'Color', 'yellow', ...
                 'FontSize', opts.FontSize, 'FontWeight', 'bold', ...
                 'BackgroundColor', 'black');
        end
        return;
    end

    %% Get colors for each detection
    colors = getColors(labels, opts.ColorMap);

    %% Draw annotations
    for i = 1:size(bboxes, 1)
        bbox = bboxes(i, :);
        score = scores(i);
        label = labels{i};
        color = colors(i, :);

        % Draw bounding box
        outputImg = insertShape(outputImg, 'Rectangle', bbox, ...
                               'Color', color * 255, ...
                               'LineWidth', opts.LineWidth);

        % Prepare label text
        if opts.ShowConfidence
            labelText = sprintf('%s: %.2f', label, score);
        else
            labelText = label;
        end

        % Calculate label position (above bbox)
        labelPos = [bbox(1), max(1, bbox(2) - 5)];

        % Draw label background
        outputImg = insertText(outputImg, labelPos, labelText, ...
                              'FontSize', opts.FontSize, ...
                              'BoxColor', color * 255, ...
                              'BoxOpacity', 0.8, ...
                              'TextColor', 'white', ...
                              'AnchorPoint', 'LeftBottom');

        % Draw orientation arrow if available
        if opts.ShowOrientation && ~isempty(orientations) && numel(orientations) >= i
            orientation = orientations(i);
            outputImg = drawOrientationArrow(outputImg, bbox, orientation, color);
        end
    end

    %% Display
    if opts.Display
        figure('Name', opts.Title, 'Position', [100, 100, 1200, 800]);
        imshow(outputImg);
        title(opts.Title, 'FontSize', 14, 'FontWeight', 'bold');

        % Add legend
        addLegend(labels, colors, scores);
    end
end

%% Helper Functions

function colors = getColors(labels, colorMapSpec)
    % Get colors for each detection based on colormap specification

    numDetections = length(labels);

    if ischar(colorMapSpec)
        switch lower(colorMapSpec)
            case 'default'
                % Use predefined colors
                baseColors = [
                    1.0, 0.0, 0.0;  % Red
                    0.0, 1.0, 0.0;  % Green
                    0.0, 0.0, 1.0;  % Blue
                    1.0, 1.0, 0.0;  % Yellow
                    1.0, 0.0, 1.0;  % Magenta
                    0.0, 1.0, 1.0;  % Cyan
                    1.0, 0.5, 0.0;  % Orange
                    0.5, 0.0, 1.0;  % Purple
                    0.0, 0.5, 0.0;  % Dark Green
                    0.5, 0.5, 0.5;  % Gray
                ];

                % Repeat colors if more detections than colors
                colors = baseColors(mod(0:numDetections-1, size(baseColors, 1)) + 1, :);

            case 'class'
                % Assign same color to same class
                uniqueLabels = unique(labels);
                numClasses = length(uniqueLabels);

                % Generate distinct colors for each class
                classColors = hsv(numClasses);

                % Map labels to colors
                colors = zeros(numDetections, 3);
                for i = 1:numDetections
                    classIdx = find(strcmp(uniqueLabels, labels{i}));
                    colors(i, :) = classColors(classIdx, :);
                end

            otherwise
                % Default rainbow colors
                colors = jet(numDetections);
        end
    elseif isnumeric(colorMapSpec)
        % Use provided color array
        colors = colorMapSpec;
    else
        % Fallback to rainbow
        colors = jet(numDetections);
    end
end

function outputImg = drawOrientationArrow(img, bbox, angle, color)
    % Draw orientation arrow on detection

    % Calculate arrow start (center of bbox)
    centerX = bbox(1) + bbox(3) / 2;
    centerY = bbox(2) + bbox(4) / 2;

    % Calculate arrow length (proportional to bbox size)
    arrowLength = min(bbox(3), bbox(4)) * 0.4;

    % Calculate arrow end point
    % Angle is in degrees, 0° is up (North), clockwise
    angleRad = deg2rad(angle - 90);  % Convert to standard math convention
    endX = centerX + arrowLength * cos(angleRad);
    endY = centerY + arrowLength * sin(angleRad);

    % Draw arrow
    outputImg = insertShape(img, 'Line', [centerX, centerY, endX, endY], ...
                           'Color', color * 255, ...
                           'LineWidth', 3);

    % Draw arrow head
    arrowHeadLength = arrowLength * 0.3;
    arrowHeadAngle = 30;  % degrees

    % Left side of arrow head
    leftAngle = angleRad - deg2rad(arrowHeadAngle);
    leftX = endX - arrowHeadLength * cos(leftAngle);
    leftY = endY - arrowHeadLength * sin(leftAngle);

    % Right side of arrow head
    rightAngle = angleRad + deg2rad(arrowHeadAngle);
    rightX = endX - arrowHeadLength * cos(rightAngle);
    rightY = endY - arrowHeadLength * sin(rightAngle);

    outputImg = insertShape(outputImg, 'Line', [endX, endY, leftX, leftY], ...
                           'Color', color * 255, ...
                           'LineWidth', 3);
    outputImg = insertShape(outputImg, 'Line', [endX, endY, rightX, rightY], ...
                           'Color', color * 255, ...
                           'LineWidth', 3);
end

function addLegend(labels, colors, scores)
    % Add legend showing detection statistics

    % Count detections per class
    uniqueLabels = unique(labels);
    legendText = cell(length(uniqueLabels), 1);

    for i = 1:length(uniqueLabels)
        className = uniqueLabels{i};
        classCount = sum(strcmp(labels, className));
        classIdx = find(strcmp(labels, className), 1);
        avgScore = mean(scores(strcmp(labels, className)));

        legendText{i} = sprintf('%s: %d (avg: %.2f)', className, classCount, avgScore);
    end

    % Display legend as text annotation
    legendStr = strjoin(legendText, '\n');

    annotation('textbox', [0.02, 0.02, 0.25, 0.15], ...
               'String', legendStr, ...
               'FitBoxToText', 'on', ...
               'BackgroundColor', 'white', ...
               'EdgeColor', 'black', ...
               'FontSize', 9);
end

%% Additional Visualization Functions

function visualizeGroundTruthComparison(img, gtBBoxes, gtLabels, predBBoxes, predLabels)
    %VISUALIZEGROUNDTRUTHCOMPARISON - Compare ground truth and predictions
    %
    % Displays ground truth in green and predictions in red for comparison

    figure('Name', 'Ground Truth vs Predictions', 'Position', [100, 100, 1200, 800]);

    % Convert to RGB if grayscale
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end

    outputImg = im2double(img);

    % Draw ground truth in green
    if ~isempty(gtBBoxes)
        for i = 1:size(gtBBoxes, 1)
            outputImg = insertShape(outputImg, 'Rectangle', gtBBoxes(i, :), ...
                                   'Color', [0, 255, 0], 'LineWidth', 2);

            labelPos = [gtBBoxes(i, 1), gtBBoxes(i, 2) - 5];
            outputImg = insertText(outputImg, labelPos, ['GT: ' gtLabels{i}], ...
                                  'FontSize', 10, 'BoxColor', 'green', ...
                                  'TextColor', 'white', 'AnchorPoint', 'LeftBottom');
        end
    end

    % Draw predictions in red
    if ~isempty(predBBoxes)
        for i = 1:size(predBBoxes, 1)
            outputImg = insertShape(outputImg, 'Rectangle', predBBoxes(i, :), ...
                                   'Color', [255, 0, 0], 'LineWidth', 2);

            labelPos = [predBBoxes(i, 1), predBBoxes(i, 2) + predBBoxes(i, 4) + 5];
            outputImg = insertText(outputImg, labelPos, ['Pred: ' predLabels{i}], ...
                                  'FontSize', 10, 'BoxColor', 'red', ...
                                  'TextColor', 'white', 'AnchorPoint', 'LeftTop');
        end
    end

    imshow(outputImg);
    title('Green: Ground Truth | Red: Predictions', 'FontSize', 14);

    % Add legend
    legend('Ground Truth', 'Predictions', 'Location', 'best');
end

function visualizeBatchDetections(images, allBBoxes, allScores, allLabels)
    %VISUALIZEBATCHDETECTIONS - Visualize multiple detections in grid
    %
    % Display multiple detection results in a grid layout

    numImages = length(images);
    gridSize = ceil(sqrt(numImages));

    figure('Name', 'Batch Detections', 'Position', [50, 50, 1400, 900]);

    for i = 1:numImages
        subplot(gridSize, gridSize, i);

        img = images{i};
        bboxes = allBBoxes{i};
        scores = allScores{i};
        labels = allLabels{i};

        % Annotate image
        outputImg = visualizeDetections(img, bboxes, scores, labels, [], ...
                                       'Display', false, 'ShowOrientation', false);

        imshow(outputImg);
        title(sprintf('Image %d (%d detections)', i, size(bboxes, 1)), 'FontSize', 10);
        axis off;
    end

    sgtitle('Batch Detection Results', 'FontSize', 14, 'FontWeight', 'bold');
end

function visualizeOrientationDistribution(orientations, labels)
    %VISUALIZEORIENTATIONDISTRIBUTION - Plot orientation distribution
    %
    % Creates polar plot showing distribution of detected orientations

    figure('Name', 'Orientation Distribution', 'Position', [100, 100, 800, 800]);

    % Convert orientations to radians
    orientationsRad = deg2rad(orientations);

    % Create polar histogram
    polarhistogram(orientationsRad, 36, 'FaceColor', 'blue', 'EdgeColor', 'black');
    title('Orientation Distribution of Detected Targets', 'FontSize', 14);

    % Add statistics
    meanOri = mean(orientations);
    stdOri = std(orientations);

    annotation('textbox', [0.15, 0.85, 0.3, 0.1], ...
               'String', sprintf('Mean: %.1f°\nStd: %.1f°', meanOri, stdOri), ...
               'FitBoxToText', 'on', ...
               'BackgroundColor', 'white', ...
               'EdgeColor', 'black', ...
               'FontSize', 11);
end

function visualizeConfidenceDistribution(scores, labels)
    %VISUALIZECONFIDENCEDISTRIBUTION - Plot confidence score distribution
    %
    % Creates histogram of detection confidence scores

    figure('Name', 'Confidence Distribution', 'Position', [100, 100, 800, 600]);

    histogram(scores, 20, 'FaceColor', 'blue', 'EdgeColor', 'black');
    xlabel('Confidence Score', 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);
    title('Distribution of Detection Confidence Scores', 'FontSize', 14);
    grid on;

    % Add statistics
    meanScore = mean(scores);
    medianScore = median(scores);
    minScore = min(scores);
    maxScore = max(scores);

    text(0.05, max(ylim) * 0.9, ...
         sprintf('Mean: %.3f\nMedian: %.3f\nMin: %.3f\nMax: %.3f', ...
                 meanScore, medianScore, minScore, maxScore), ...
         'FontSize', 11, 'BackgroundColor', 'white', 'EdgeColor', 'black');
end
