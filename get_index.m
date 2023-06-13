% Get period index range
function index = get_index(num_domain)
index = {};
for i = 1:num_domain
    for j = i + 1:num_domain
        index{end + 1} = [i, j];
    end
end
end