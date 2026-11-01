function x = rom_lsq(A, b, method)
% ROM_LSQ  Least-squares solution x = argmin ||A x - b|| of the small dense
% problems of the reduced iteration (LSPG state, Gauss-Newton increment,
% adjoint, enriched adjoint of the dual-weighted estimate).
%   method = 'qr'     : Householder QR (MATLAB backslash on the tall matrix),
%                       the default;
%   method = 'normal' : the normal equations (A'A) x = A'b, the way the runs of
%                       Sections 5.1-5.3 of the paper were computed. They square
%                       the condition number of A, which shows in the cannula
%                       objective at the level of 1e-6.
switch method
    case 'qr',     x = A \ b;
    case 'normal', x = (A.'*A) \ (A.'*b);
    otherwise,     error('rom_lsq: unknown method %s', method);
end
end
