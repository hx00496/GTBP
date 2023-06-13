%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% predict the location of the next centre point
function [PopCenter,PopManifold,Pop] = GTBP(mop,Pop,PopSize,DimSize,ObjNum,LowerUpper,PopCenter,PopManifold,Center_archive)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 1:Manifold
num_seq     =size(Center_archive,1);
CenterPoint =Center_archive(num_seq,:);
Manifold    = Pop(:,1:DimSize) - ones(size(Pop,1),1)*CenterPoint(1,1:DimSize);
% save shape points C1 C2
if (num_seq < 300)
    PopManifold(1).C = Manifold;
elseif (num_seq >= 300)
    PopManifold(2).C = PopManifold(1).C;
    PopManifold(1).C = Manifold;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 2: if the history info. is not enough for prediction, then use random initialization
% randomly sample half of Pt and randomly select the other half from Pt-1
if num_seq < 300
    Pop = RIS(mop,1,Pop,PopSize, LowerUpper, ObjNum, DimSize);
else
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Step 3: find the parent point for each point in C1
C1 = PopManifold(1).C;
C2 = PopManifold(2).C;
for i = 1 : PopSize
    pindex(i) = 0;   dismin = 1.0E100;
    for j = 1 : PopSize
        dis = 0.0;
        for k = 1 : DimSize
            dis = dis + (C1(i,k) - C2(j,k))^2;
        end
        if (dis<dismin)
            dismin = dis;  pindex(i) = j;
        end
    end
end
for k = 1 : DimSize
    CStd(k) = 0.0;
    for i = 1 : PopSize
        CStd(k) = CStd(k) + (C1(i,k) - C2(pindex(i),k))^2;
    end
    CStd(k) = CStd(k)/(PopSize);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: predict the center point
[PredictCenter,RMSE]=training_progress(Center_archive,DimSize);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: sample new trial solutions
OldPop = Pop;
for j = 1 : DimSize
    Std(j) = sqrt(RMSE^2 + CStd(j));
    for i = 1 : PopSize
        Pop(i,j) = 	PredictCenter(j) + C1(i,j) + normrnd(0,Std(j));
        if Pop(i,j) < LowerUpper(1,j)
            Pop(i,j) = 0.5*(OldPop(pindex(i),j) + LowerUpper(1,j));
        elseif Pop(i,j) > LowerUpper(2,j)
            Pop(i,j) = 0.5*(OldPop(pindex(i),j) + LowerUpper(2,j));
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

