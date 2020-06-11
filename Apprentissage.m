pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

imgDir = fullfile('.','dataset','images_prepped_train_matlab');
imds = imageDatastore(imgDir);

I = readimage(imds,27);
imshow(I)

classes = [ %nos 4 classes
    "Trottoir"
    "Route"
    "Obstacle"
    "PassagePieton"
    ];

labelIDs = camvidPixelLabelIDs();
labelDir = fullfile('.','dataset','annotations_prepped_train_matlab');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [256 640 3];

C = readimage(pxds,27);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
%imshow(B)
pixelLabelColorbar(cmap,classes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Apprentissage

tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);

numTrainingImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestingImages = numel(imdsTest.Files)

% Specify the number of classes.
numClasses = numel(classes);

% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);

[net, info] = trainNetwork(pximds,lgraph,options);

Dir = fullfile('.','dataset','images_prepped_test');
imdsj = imageDatastore(Dir);
I = readimage(imdsj,31);

C = semanticseg(I, net);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function labelIDs = camvidPixelLabelIDs()
% Return the label IDs corresponding to each class.
%
% The CamVid dataset has 32 classes. Group them into 11 classes following
% the original SegNet training methodology [1].
%
% The 11 classes are:
%   "Sky" "Building", "Pole", "Road", "Pavement", "Tree", "SignSymbol",
%   "Fence", "Car", "Pedestrian",  and "Bicyclist".
%
% CamVid pixel label IDs are provided as RGB color values. Group them into
% 11 classes and return them as a cell array of M-by-3 matrices. The
% original CamVid class names are listed alongside each RGB value. Note
% that the Other/Void class are excluded below.
labelIDs = { ...
    
    % "Trotoir"
    [
    128 128 000; ... % "Trotoir"
    ]
    
    % "Route"
    [
    128 64 128; ... % "Route"
    ]
    
    % "Obstacle"
    [
    128 000 000; ... % "Obstacle"
    ]

    % "PassagePieton"
    [
    064 128 192; ... % "PassagePieton"
    ]
};
end

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end

function cmap = camvidColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    128 128 000   % Trotoir
    128 64 128     % Route
    000 000 128     % "Rails"
    128 000 000      % Obstacle
    064 128 192       % PassagePieton
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(0.20 * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = camvidPixelLabelIDs();

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

function B = labeloverlay(varargin)
%LABELOVERLAY Overlay label matrix regions on a 2-D image.
%   B = LABELOVERLAY(A,L) fills the input image with a different solid
%   color for each label in the label matrix L. L must be a valid MxN label matrix
%   that agrees with the size of A.
%
%   B = LABELOVERLAY(A,BW) fills the input image with a solid color where
%   BW is true. BW must be a valid mask that agrees with the size of A.
%
%   B = LABELOVERLAY(A,C) fills in the input image with a different solid
%   color for each label specified by the categorical matrix C.
%
%   B = LABELOVERLAY(___,NAME,VALUE) computes the fused overlay image B
%   using NAME/VALUE parameters to control aspects of the computation.
%
%   Parameters include:
%
%   'Colormap'              Mx3 colormap where M is the number of labels in
%                           the label matrix L or binary mask BW. RGB
%                           triplets in each row of the colormap must be
%                           normalized to the range [0,1]. A string or
%                           character vector corresponding to one of the
%                           valid inputs to the MATLAB colormap function is
%                           also accepted, in which case a permuted form of
%                           the specified colormap suitable for labeled
%                           region visualization will be used.
%
%                           Default: 'jet'
%
%   'IncludedLabels'        Scalar or vector of integer values in the range
%                           [0,max(L(:))] that specify the labels that will
%                           be falsecolored and blended with the input
%                           image. When a categorical, C, is provided as
%                           the specification of the labeled regions,
%                           'IncludedLabels' can also be a vector of
%                           strings corresponding to labels in C.
%
%                           Default: 1:length(L(:))
%
%   'Transparency'          Scalar numeric value in the range [0,1] that
%                           controls the blending of the label matrix with
%                           the original input image A. A value of 1.0
%                           makes the label matrix coloring completely
%                           transparent. A value of 0.0 makes the label
%                           matrix coloring completely opaque.
%
%                           Default: 0.5
%
%   Class Support
%   -------------
%   The input image A is of type uint8, uint16, single,
%   double, logical, or int16. The input label matrix L is a numeric
%   matrix. B is an RGB image of type uint8.
%
%   Example 1 - Visualize over-segmentation of RGB data
%   ---------
%    A = imread('kobi.png');
%    [L,N] = superpixels(A,20);
%    figure
%    imshow(labeloverlay(A,L));
%
%   Example 2 - Visualize binary-segmentation of greyscale image
%   ---------
%   A = imread('coins.png');
%   t = graythresh(A);
%   BW = imbinarize(A,t);
%   figure
%   imshow(labeloverlay(A,BW))
%
%   Example 3 - Visualize segmentation specified as categorical array
%   ---------
%   A = imread('coins.png');
%   t = graythresh(A);
%   BW = imbinarize(A,t);
%   stringArray = repmat("table",size(BW));
%   stringArray(BW) = "coin";
%   categoricalSegmentation = categorical(stringArray);
%   figure
%   imshow(labeloverlay(A,categoricalSegmentation,'IncludedLabels',"coin"));
%
%   See also superpixels, imoverlay

%   Copyright 2016-2019 The MathWorks, Inc.

narginchk(2,inf);

parsedInputs = parseInputs(varargin{:});

% Cast and passthrough if labels are all zeros
if isempty(parsedInputs.IncludedLabels)
    B = im2uint8(parsedInputs.A);
    return;
end

A = parsedInputs.A;
L = parsedInputs.L;
cmap = parsedInputs.Colormap;
includeList = parsedInputs.IncludedLabels;
alpha = 1-parsedInputs.Transparency;

B = images.internal.labeloverlayalgo(A,L,cmap,alpha,includeList);

B = im2uint8(B);

end

function cmapOut = formPermutedColormap(cmap)
% Create run-to-run reproducible shuffled version of the specified
% colormap. When viewing labeled regions, you don't want nearby regions to
% have similar colors. Many of the built-in colormaps take a path through
% some colorspace, so nearby elements in colormaps tend to have similar
% colors, which we don't want.

s = rng;
c = onCleanup(@() rng(s));
rng('default');
totalLabels = size(cmap,1);
cmapOut = cmap(randperm(totalLabels),:);

end

function results = parseInputs(varargin)

A = varargin{1};
L = varargin{2};

allowedTypes = images.internal.iptnumerictypes();
allowedTypes{end+1} = 'logical';
allowedTypes{end+1} = 'categorical';
validateattributes(A,{'single', 'double', 'uint8', 'uint16', 'int16'},{'nonsparse','real','nonempty'},mfilename,'A');
isGrayOrRGBImage = ismatrix(A) || ((ndims(A) == 3) && (size(A,3) == 3));
if ~isGrayOrRGBImage
    error(message('images:labeloverlay:inputImageMustBeGrayOrRGB'));
end

validateattributes(L,allowedTypes,{}); % Just do type checking to start.

if ~iscategorical(L)
    Ldouble = double(L);
    % Add 1 to accomodate for Label 0
    maxLabel = max(Ldouble(:));
    totalLabels = maxLabel + 1;
else
    Ldouble = double(uint32(L));
    totalLabels = length(categories(L));
    maxLabel = length(categories(L));
end

validateattributes(Ldouble,allowedTypes,{'integer','nonsparse','real','nonnegative','nonempty','ndims',2},mfilename);

A = im2single(A);

% Function scoped variables used in input parsing
cmapFunctionScope = formPermutedColormap(jet(totalLabels));

parser = inputParser();
parser.addParameter('Transparency',0.5,@validateTransparency);
parser.addParameter('IncludedLabels',1:maxLabel,@validateIncludedLabels);
parser.addParameter('Colormap',cmapFunctionScope,@validateColormap);

parser.parse(varargin{3:end})

results = parser.Results;
results.IncludedLabels = postProcessIncludedLabels(L,results.IncludedLabels);

results.A = A;
results.L = Ldouble;
results.MaxLabel = maxLabel;
results.IncludedLabels = double(results.IncludedLabels);
results.Colormap = single(cmapFunctionScope);
results.Transparency = single(results.Transparency);

if size(results.Colormap,1) < length(results.IncludedLabels)
    error(message('images:labeloverlay:badColormap'));
end

if (max(results.IncludedLabels(:)) > maxLabel) 
   error(message('images:labeloverlay:badBackgroundLabel')); 
end

sizeA = size(A);
if ~isequal(size(L), sizeA(1:2))
   error(message('images:labeloverlay:inputImageSizesDisagree'));
end

    function TF = validateTransparency(transparency)
        validateattributes(transparency,{'single','double'},{'real','scalar','nonsparse','<=',1,'>=',0},mfilename,'Transparency');
        TF = true;
    end

    function TF = validateColormap(cmap)
        if isnumeric(cmap)
            validateattributes(cmap,{'single','double'},{'real','2d','nonsparse','ncols',3,'<=',1,'>=',0},mfilename,'Colormap');
            cmapFunctionScope = cmap;
        else
            try
               cmapTemp = feval(cmap,totalLabels);
               cmapFunctionScope = formPermutedColormap(cmapTemp);
            catch
                error(message('images:labeloverlay:invalidColormapString'));
            end
        end
        TF = true;
    end

    function TF = validateIncludedLabels(includedLabels)
        
        validTypes = images.internal.iptnumerictypes();
        if isnumeric(includedLabels)
            validateattributes(includedLabels,validTypes,...
                {'real','vector','nonsparse','integer','nonnegative'},mfilename,'IncludedLabels');
        end
        
        TF = true;
    end

end

function includedLabels = postProcessIncludedLabels(C,includedLabels)

if iscategorical(C) && isstring(includedLabels)
    if all(iscategory(C,includedLabels))
        categoriesSet = string(categories(C));
        includedLabelsOut = zeros([1 length(includedLabels)]);
        for i = 1:length(includedLabels)
           includedLabelsOut(i) = find(includedLabels(i) == categoriesSet);
        end
        includedLabels = includedLabelsOut;
    else
        error(message('images:labeloverlay:invalidCategory')); 
    end
end

end