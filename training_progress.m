function  [wait_data_pred,RMSE]=training_progress(Center_archive,DimSize)
% Import the original sequence data
data =Center_archive;

% Define the lengths of training、validation and test sets 
numTimeStepsTrain = floor(0.8 * size(data, 1));
numTimeStepsvaild = floor(0.9 * size(data, 1));
dataTrain = data(1:numTimeStepsTrain, :);
datavaild = data(numTimeStepsTrain + 1:numTimeStepsvaild, :);
dataTest = data(numTimeStepsvaild + 1:end, :);

% Generate training, validation, and test data
[XTrain, YTrain] = Data_generator(dataTrain);
[XVaild, YVaild] = Data_generator(datavaild);
[XTest, YTest] = Data_generator(dataTest);

% Define the layer structure
layerAll = [
    sequenceInputLayer(DimSize, "Name", "input")
    gruLayer(64, "Name", "gru_1")
    gruLayer(64, "Name", "gru_2", "OutputMode", "last")
    fullyConnectedLayer(64, "Name", "fc_1")
    fullyConnectedLayer(64, "Name", "fc_2")
    batchNormalizationLayer("Name", "batchnorm")
    reluLayer("Name", "relu")
    dropoutLayer(0.5, "Name", "dropout")
    fullyConnectedLayer(DimSize, "Name", "fc_3")
    ];

% Define layer objects
All = layerGraph(layerAll);

% Generate layer objects
dlAll = dlnetwork(All);

% Define the loss function
lossType = "mse";

% Split the training set into segments
num_domain = 3;
dis_type = "coral";
res = TDC(num_domain, XTrain, dis_type);

% Generate training sets for each segment
train_listX = cell(1, length(res));
train_listY = cell(1, length(res));
for i = 1:length(res)
    train_listX{i} = XTrain(1, res{1, i}(1, 1):res{1, i}(1, 2));
    train_listY{i} = YTrain(res{1, i}(1, 1):res{1, i}(1, 2), :);
end

% Get the indices
index = get_index(num_domain);

% Determine the number of batches in the smallest segment
MiniBatchSize=36;
len_loader=Inf;
for i=1:length(train_listX)
    if length(train_listX{1,i}) < len_loader
        len_loader=length(train_listX{1,i});
    end
end
num_train=ceil(len_loader/MiniBatchSize);

% Initialize variables
numEpochs = 50;
learnRate = 0.1;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
trailingAvgG = [];
trailingAvgSqG = [];
trailingAvg = [];
trailingAvgSqD = [];

validationLossCount = 0;  % Counter to track the number of times validation loss exceeds training loss

% Training loop
for epoch = 1:numEpochs
    for i = 1:length(index)
        % Generate and load mini-batch data
        for k=1:length(train_listX)
            dsX1{k} = arrayDatastore(permute(reshape(cell2mat(train_listX{1,k}),20,24,[]),[3,1,2]),"ReadSize",36);
            dsY1{k} = arrayDatastore(train_listY{1,k},"ReadSize",36);
            dsTrain{k} = combine(dsX1{k},dsY1{k});
            mbq{k} = minibatchqueue(dsTrain{k},...
                'MiniBatchSize',36,...
                'PartialMiniBatch','return');
        end
        % Initialize mini-batch counts for the two periods
        numData1=0;
        numData2=0;
        % Convert mini-batch data to arrays
        while hasdata(mbq{1,index{i}(1)})
            mbq1 = mbq{1,index{i}(1)};
            [X1,Y1]=next(mbq1);
            minibatch1{numData1+1}={X1,Y1};
            numData1 = numData1 + size(minibatch1, 1);
        end
        while hasdata(mbq{1,index{i}(2)})
            mbq2 = mbq{index{i}(2)};
            [X2,Y2]=next(mbq2);
            minibatch2{numData2+1} = {X2,Y2};
            numData2 = numData2 + size(minibatch1, 1);
        end
        % Batch iteration
        while numData1>=2 && numData2>=2
            % Generate random numbers
            randomBatchIndex1 = randi([1, numData1-1]);
            randomBatchIndex2 = randi([1, numData2-1]);
            % Retrieve a random mini-batch
            X1 = minibatch1{randomBatchIndex1}{1};
            Y1 = minibatch1{randomBatchIndex1}{2};
            X2 = minibatch2{randomBatchIndex2}{1};
            Y2 = minibatch2{randomBatchIndex2}{2};
            % Check if the batches are consistent
            if size(X1, 4) == size(X2, 4)
                % Remove the batch from the arrays
                minibatch1(randomBatchIndex1) = [];
                minibatch2(randomBatchIndex2) = [];
                numData1=numData1-1;
                numData2=numData2-1;
                X1=X1.squeeze;
                X2=X2.squeeze;
                % Concatenate features ,labels
                feat_all=cat(3,X1,X2);
                label_all=cat(1,Y1,Y2)';
                % Gradients initialization
                gradients = dlarray(zeros(size(dlAll.Learnables), "single"), "SSCB");
                % Forward pass
                feat_all = dlarray(feat_all, "CTB");
                label_all = dlarray(label_all, "CTB");
                [dlYPred, state] = dlAll.forward(feat_all);
                % Compute the loss % Backward pass
                [total_loss, gradients, ~] = dlfeval(@modelLoss, dlAll, feat_all, dlYPred, label_all, state);
                % Update the network parameters using ADAM optimizer 
                dlAll = adamupdate(dlAll, gradients, trailingAvgG, trailingAvgSqG, epoch, learnRate, gradientDecayFactor, squaredGradientDecayFactor); 
            end
        end
    end
    % Compute the validation loss after each epoch
    validationLoss = computeValidationLoss(dlAll, XVaild, YVaild);
    % Check if the validation loss exceeds the training loss for 5 consecutive times
    if validationLoss > total_loss
        validationLossCount = validationLossCount + 1;
    else
        validationLossCount = 0;
    end
    % If validation loss exceeds training loss for 5 consecutive times, stop training
    if validationLossCount >= 5
        disp("Validation loss exceeded training loss for 5 consecutive times. Stopping training.");
        break;
    end
    
end
% Predict the test data
YPred=[];
numTimeStepsTest = numel(XTest);
dlAll = resetState(dlAll);
[dlAll,Ypred] = predictAndUpdateState(dlAll,XTest);

% Calculate the root mean squared error (RMSE)
for i = 1:size(YTest,1)
    rmse(i) = sqrt(mean((Ypred(i,:) - YTest(i,:)).^2,"all"));
end
RMSE=sum(rmse)/size(YTest,1);
wait_data=XTest(1,size(XTest,2));
% Initialize network state & prediction
dlAll = resetState(dlAll);
[~,wait_data_pred] = predictAndUpdateState(dlAll,wait_data);
end
