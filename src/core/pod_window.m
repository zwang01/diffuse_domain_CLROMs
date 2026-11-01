function [Snap, Phi, r] = pod_window(Snap, new, w_max, r_max, pod_tol)
% POD_WINDOW  Window update of Algorithm 2: append the columns of `new` (each
% normalized to unit length; a snapshot, possibly with its predictor
% direction) to the window Snap, drop the oldest columns beyond w_max, and
% return the POD basis Phi of the window, truncated at the relative singular
% value pod_tol and capped at r_max modes. new = [] only recomputes the basis.
for j = 1:size(new, 2)
    nj = norm(new(:,j));
    if nj > 0, new(:,j) = new(:,j)/nj; end
end
if ~isempty(new)
    Snap = [Snap, new];
    if size(Snap, 2) > w_max, Snap = Snap(:, end-w_max+1:end); end
end
if isempty(Snap), Phi = zeros(size(Snap,1), 0); r = 0; return, end
[U0, Sv, ~] = svd(Snap, 'econ');
s = diag(Sv);
r = min(r_max, max(1, sum(s >= pod_tol*s(1))));
Phi = U0(:, 1:r);
end
