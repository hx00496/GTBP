function [XTrain,YTrain]=Data_generator(dataTrain)
% data normalization
% mu = mean(dataTrain);
% sig = std(dataTrain);
% for i=1:size(dataTrain,1)
%     for j=1:size(dataTest,2)
%         dataTrainStandardized(i,j) = (dataTrain(i,j) - mu(1,j)) / sig(1,j);
%     end
% end
dataTrainStandardized= dataTrain';
seq_length=24;
XTrain={};
YTrain={};
for ii =1:size(dataTrain,1)-seq_length
    XTrain{ii}=dataTrainStandardized(:,ii:ii+seq_length-1);
    YTrain{ii}=dataTrainStandardized(:,ii+seq_length);
end
YTrain=cell2mat(YTrain)';
end