function F = fact_chol(A)
% FACT_CHOL  Sparse Cholesky factorization of a symmetric positive definite
% matrix with the approximate minimum degree ordering, returned as solve
% closures F.solve = F.solvet ~ A \ x (fact_lu has the same interface).
%
% The ordering matters: chol(A) with one output does not reorder, and on the
% 200 x 100 cantilever grid the factor of the lexicographically numbered
% stiffness matrix has 1.6e7 nonzeros against 3.2e6 with AMD, and takes four
% times as long.
% Counts one factorization (ddm_count_fact).
ddm_count_fact;
[R, flag, p] = chol(A, 'vector');
if flag ~= 0, error('fact_chol: matrix not positive definite (flag %d)', flag); end
F.solve  = @(x) apply(R, p, x);
F.solvet = F.solve;
F.n = size(A, 1);
F.kind = 'chol';
end

function y = apply(R, p, x)
y = zeros(size(x));
y(p, :) = R \ (R.' \ x(p, :));
end
