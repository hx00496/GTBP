function [total_loss, gradients, state] = modelLoss(dlAll,feat_all, dlYPred, label_all, state)
% Transfer loss parameters
dw = 0.5;
trans_loss = "adv";


% Get output of gru1 layer
outputLayerName1 = 'gru_1'; % Name of the target layer 1
% Find all layers before the target layer
layerNames = {dlAll.Layers.Name};
targetLayerIndex = find(strcmp(layerNames, outputLayerName1));
layersBeforeTarget = dlAll.Layers(1:targetLayerIndex);
% Create a new dlnetwork object
dlNetworkBeforeTarget = dlnetwork(layersBeforeTarget);
% Perform forward propagation using the new dlnetwork object
[dlYPred1,~] = dlNetworkBeforeTarget.forward(feat_all);
% Get the output of the target layer
out_list1 = dlYPred1;

% Get output of gru2 layer
outputLayerName2 = 'gru_2'; % Name of the target layer 2
% Find all layers before the target layer
layerNames = {dlAll.Layers.Name};
targetLayerIndex2 = find(strcmp(layerNames, outputLayerName2));
layersBeforeTarget2 = dlAll.Layers(1:targetLayerIndex2);
% Create a new dlnetwork object
dlNetworkBeforeTarget2 = dlnetwork(layersBeforeTarget2);
% Perform forward propagation using the new dlnetwork object
[dlYPred2, ~] = dlNetworkBeforeTarget2.forward(feat_all);
% Get the output of the target layer
out_list2 = dlYPred2;


% Calculate transfer loss
% Merge outputs, merge weights
out_list = {out_list1, out_list2};
out_list2_extended = repmat(out_list2, [1 1 24]);
% Calculate transfer loss
len_win = 0;
len_seq = 24;
trans_loss = "adv";
loss_transfer = 0;
for i = 1:length(out_list)
    criterion_transder = TransferLoss(trans_loss, size(out_list{1, i}, 3));
    h_start = 1;
    for j = h_start:len_seq
        loss_transfer = loss_transfer + criterion_transder.compute(out_list1(:, :, j), out_list2_extended(:, :, j));
    end
end

% Calculate prediction loss
label_all=squeeze(label_all);
label_all = dlarray(label_all, "CT");
dlYPred =  dlarray(dlYPred, "CT");
loss_s_t = mse(dlYPred, label_all);
L1loss = mean(abs(label_all - dlYPred), 2);

% Total loss
total_loss = loss_s_t + dw * loss_transfer;

% Calculate gradients
gradients = dlgradient(total_loss, dlAll.Learnables);
end