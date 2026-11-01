function [x, flag, relres, it] = pc_cg(A, b, x0, Mfun, tol, maxit)
% PC_CG  Minimal preconditioned conjugate gradients for a symmetric positive
% definite A (sparse matrix or function handle) with a symmetric positive
% definite preconditioner M^{-1} = Mfun (here: the triangular solves of the
% retained Cholesky factorization K_ref). Stops when the preconditioned
% residual satisfies ||M^{-1}(b - A x)|| <= tol*||M^{-1} b||, the criterion of
% pc_gmres, so that the fallback tolerances of the three problems mean the
% same thing. flag = 0 on convergence within maxit iterations. MATLAB's pcg
% carries a per-call overhead that this stripped version does not.
if isa(A, 'function_handle'), Afun = A; else, Afun = @(v) A*v; end
bnorm = max(norm(Mfun(b)), realmin);
if isempty(x0), x = zeros(size(b)); r = b; else, x = x0; r = b - Afun(x); end
z = Mfun(r); p = z; rz = r.'*z;
flag = 1; it = 0; relres = norm(z)/bnorm;
if relres <= tol, flag = 0; return, end
for j = 1:maxit
    Ap = Afun(p);
    alpha = rz/(p.'*Ap);
    x = x + alpha*p;
    r = r - alpha*Ap;
    z = Mfun(r);
    it = j; relres = norm(z)/bnorm;
    if relres <= tol, flag = 0; return, end
    rz_new = r.'*z;
    p = z + (rz_new/rz)*p;
    rz = rz_new;
end
end
