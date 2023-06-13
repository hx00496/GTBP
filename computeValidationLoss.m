% Validation Set Loss Function
function loss = computeValidationLoss(dlAll, XVaild, YVaild)
loss=0;
XVaild_listX = cell(1, 1);
XVaild_listY = cell(1, 1);
for i = 1:1
    XVaild_listX{i} = XVaild;
    XVaild_listY{i} = YVaild;
end

for i=1:1
    dsX1{i} = arrayDatastore(permute(reshape(cell2mat(XVaild_listX{1,i}),20,24,[]),[3,1,2]),"ReadSize",10);
    dsY1{i} = arrayDatastore(XVaild_listY{1,i},"ReadSize",10);
    dsTrain{i} = combine(dsX1{i},dsY1{i});
    mbq{i} = minibatchqueue(dsTrain{i},...
        'MiniBatchSize',36,...
        'PartialMiniBatch','return');
end

mbq1 = mbq{1};
while hasdata(mbq1)
    [X1, Y1] = next(mbq1);
    X1=X1.squeeze;
    feat_all = dlarray(X1, "CTB");
    
    % Perform forward propagation of the network
    dlYPred = dlAll.forward(feat_all);
    
    % Calculate the loss
    dlYPred = extractdata(dlYPred);
    Y1 = extractdata(Y1);
    % Adjust dimensions
    loss = loss+mse(dlYPred, Y1');
end
end