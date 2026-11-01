function [x, flag, relres, it] = pc_gmres(A, b, x0, Mfun, tol, maxit)
% PC_GMRES  Minimal unrestarted left-preconditioned GMRES.
% Solves A x = b to ||M^{-1}(b - A x)|| <= tol*||M^{-1} b|| by the Arnoldi
% process with modified Gram-Schmidt and Givens rotations, warm-started from
% x0 (empty = zero). A is a sparse matrix or a function handle; Mfun applies
% M^{-1} (here: the triangular solves of the retained factorization K_ref).
% flag = 0 on convergence within maxit iterations.
%
% MATLAB's gmres carries a fixed overhead of several tens of milliseconds per
% call, comparable to a whole fallback solve of a few iterations at the grid
% sizes of the paper; this stripped version has none.
if isa(A, 'function_handle'), Afun = A; else, Afun = @(v) A*v; end
n = numel(b);
bnorm = max(norm(Mfun(b)), realmin);
if isempty(x0), x = zeros(n, 1); r = b; else, x = x0; r = b - Afun(x); end
z = Mfun(r); beta = norm(z);
flag = 1; it = 0; relres = beta/bnorm;
if relres <= tol, flag = 0; return, end

V = zeros(n, maxit+1); H = zeros(maxit+1, maxit);
cs = zeros(maxit, 1); sn = zeros(maxit, 1); g = zeros(maxit+1, 1);
V(:,1) = z/beta; g(1) = beta;
for j = 1:maxit
    w = Mfun(Afun(V(:,j)));
    for i = 1:j                                  % modified Gram-Schmidt
        H(i,j) = V(:,i)'*w;
        w = w - H(i,j)*V(:,i);
    end
    H(j+1,j) = norm(w);
    if H(j+1,j) > 0, V(:,j+1) = w/H(j+1,j); end
    for i = 1:j-1                                % previous rotations
        t        =  cs(i)*H(i,j) + sn(i)*H(i+1,j);
        H(i+1,j) = -sn(i)*H(i,j) + cs(i)*H(i+1,j);
        H(i,j)   = t;
    end
    den = hypot(H(j,j), H(j+1,j));               % new rotation
    if den == 0, den = realmin; end
    cs(j) = H(j,j)/den; sn(j) = H(j+1,j)/den;
    H(j,j) = den; H(j+1,j) = 0;
    g(j+1) = -sn(j)*g(j); g(j) = cs(j)*g(j);
    it = j; relres = abs(g(j+1))/bnorm;
    if relres <= tol, flag = 0; break, end
end
y = H(1:it,1:it) \ g(1:it);
x = x + V(:,1:it)*y;
end
