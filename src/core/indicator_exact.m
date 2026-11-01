function [eta_s, eta_a] = indicator_exact(rs, ra, dJdU, Kref, Unorm)
% INDICATOR_EXACT  Preconditioned error indicators of Algorithm 2,
%   eta_s = ||K_ref^{-1} r_s|| / ||U_ref||,
%   eta_a = ||K_ref^{-T} r_a|| / ||K_ref^{-T} dJ/dU||,
% with K_ref the retained factorization (struct of solve closures, fact_lu /
% fact_chol) and Unorm the norm of the full-order state that goes with it.
% K_ref^{-1} r_s is the error of the reduced state up to the change of the
% operator since K_ref was formed, so eta_s estimates the relative state error.
% For a self-adjoint problem (ra empty: the adjoint is the state itself)
% eta_a = 0 and only the state indicator is evaluated.
eta_s = norm(Kref.solve(rs)) / Unorm;
if isempty(ra)
    eta_a = 0;
else
    eta_a = norm(Kref.solvet(ra)) / max(norm(Kref.solvet(dJdU)), 1e-30);
end
end
