% Domain Loss Function
function loss= CORAL(source, target)
    source=source';
    target=target';
    d = size(source,2);
    ns = size(source,1);
    nt = size(target,1);

    % source covariance
    tmp_s = ones(1, ns) * source;
    cs = (source' * source - (tmp_s' * tmp_s) / ns) / (ns - 1);

    % target covariance
    tmp_t = ones(1, nt) * target;
    ct = (target' * target - (tmp_t' * tmp_t) / nt) / (nt - 1);

    % frobenius norm
    loss = sum(sum((cs - ct) .^ 2));
    loss = loss / (4 * d * d);
end