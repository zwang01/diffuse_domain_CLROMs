function [x, ok, nref] = fact_solve(F, A, b, transp, tol, maxref)
% FACT_SOLVE  Solve A x = b (transp = false) or A.' x = b (transp = true) to a
% relative residual tol with the factorization F of A (fact_lu / fact_chol),
% by iterative refinement x <- x + F^{-1}(b - A x). Exact factors converge in
% one step; the low-fill LU of the flow problem in two or three. ok = false
% when tol is not reached within maxref steps.
if nargin < 5 || isempty(tol), tol = 1e-12; end
if nargin < 6 || isempty(maxref), maxref = 8; end
nb = norm(b);
if nb == 0, x = zeros(size(b)); ok = true; nref = 0; return, end
if transp, Fs = F.solvet; else, Fs = F.solve; end
x = Fs(b); ok = false;
for nref = 1:maxref
    if transp, r = b - A.'*x; else, r = b - A*x; end
    if norm(r) <= tol*nb, ok = true; return, end
    x = x + Fs(r);
end
if transp, r = b - A.'*x; else, r = b - A*x; end
ok = norm(r) <= tol*nb;
end
