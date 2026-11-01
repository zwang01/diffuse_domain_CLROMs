function F = fact_lu(A, thresh)
% FACT_LU  Sparse LU factorization returned as a struct of solve closures,
%     F.solve(x)  ~ A  \ x,     F.solvet(x) ~ A.' \ x,
% from the explicit factors [L,U,P,Q,R] = lu(A, thresh), P*(R\A)*Q = L*U, whose
% triangular solves are cheaper than those of a decomposition object; the
% factorization is retained as the preconditioner K_ref of Algorithm 2.
%
% thresh = []        : UMFPACK's default pivoting (scalar problem).
% thresh = [0.1 0]   : diagonal pivots accepted (phi-weighted flow). The saddle
%   matrix carries a uniform pressure nugget on its diagonal; UMFPACK's
%   symmetric strategy rejects every nugget pivot and produces ~80x nnz(A)
%   fill. Accepting the diagonal pivots brings the fill to ~6x and the time
%   down thirtyfold; the factors are then inexact (relative residual ~1e-5),
%   and FACT_SOLVE removes the error by iterative refinement. As a
%   preconditioner the inexact factors are as good as exact ones.
%
% Every call counts one factorization (ddm_count_fact).
ddm_count_fact;
if nargin < 2 || isempty(thresh)
    [L, U, P, Q, R] = lu(A);
else
    [L, U, P, Q, R] = lu(A, thresh);
end
F.solve  = @(x) apply(L, U, P, Q, R, x, false);
F.solvet = @(x) apply(L, U, P, Q, R, x, true);
F.n = size(A, 1);
F.kind = 'lu';
end

function y = apply(L, U, P, Q, R, x, transp)
ws = warning('off', 'MATLAB:nearlySingularMatrix');   % tiny nugget pivots
warning('off', 'MATLAB:singularMatrix');
if transp                                             % A' = Q*U'*L'*P*R
    y = R \ (P.' * (L.' \ (U.' \ (Q.' * x))));
else                                                  % A  = R*P'*L*U*Q'
    y = Q * (U \ (L \ (P * (R \ x))));
end
warning(ws);
end
