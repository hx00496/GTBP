%% TDC
function res= TDC(num_domain, XTrain, dis_type)
% Start index
start_time = 1;
% Number of splits
split_N = 10;
num_day = size(XTrain,2);
% Convert to array
feat=cell2mat(XTrain);
feat_shape_1 = size(XTrain{1,1},2) ;
% Default to split into ten segments
selected = [0, 10];
candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9];
if ismember(num_domain,[2, 3, 5, 7, 10])
    % If splitting into num_domain, then insert num_domain-1 breakpoints
    while length(selected) - 2 < num_domain - 1
        % Initialize distances
        distance_list = [];
        start = 1;
        % Iterate through candidate points and find the one with maximum dissimilarity
        for can = candidate
            selected=[selected,can];
            selected=sort(selected,2);
            dis_temp = 0;
            for  i = 2: length(selected)-1
                for j = i: length(selected)-1
                    index_part1_start = start + floor(selected(1,i-1) / split_N * num_day)* feat_shape_1;
                    index_part1_end = start + floor(selected(1,i) / split_N * num_day) * feat_shape_1-1;
                    feat_part1 = feat(:,index_part1_start: index_part1_end);
                    index_part2_start = start + floor(selected(1,j) / split_N * num_day) * feat_shape_1;
                    index_part2_end = start + floor(selected(1,j+1) / split_N * num_day) * feat_shape_1-1;
                    feat_part2 = feat(:,index_part2_start:index_part2_end);
                    criterion_transder =TransferLoss(dis_type, size(feat_part1,1));
                    dis_temp = dis_temp+criterion_transder.compute(feat_part1, feat_part2);
                end
            end
            distance_list=[distance_list,dis_temp];
            selected = selected(~ismember(selected,can));    
        end
        [~,can_index1] =max(distance_list);
        can_index=candidate(can_index1);
        selected=[selected,can_index];
        selected=sort(selected,2);
        candidate(can_index1) =[];
    end
    res=cell(1,length(selected)-1);
    % Format the segmentation results and return
    for i =1:length(selected)-1
        if i == 1
            sel_start_time = start_time;
        else
            sel_start_time = floor(num_day / split_N * selected(:,i))+1;
        end
        sel_end_time = floor(num_day / split_N * selected(:,i+1));
        res(1,i)={[sel_start_time sel_end_time]};
    end
end
end
