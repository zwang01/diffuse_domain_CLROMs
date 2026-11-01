function [phi, info] = ac_update_vol(phi, sens, freeze, phiFrozen, Vcells, ac)
% Sensitivity-driven Allen-Cahn design update with an EXACT volume
% constraint via a Lagrange multiplier, following step4_AC.m of the Poisson
% code. Each of ac.Kin inner steps solves the semi-implicit system
%   (I - tau*kappa*L) phi_new = phi + tau*(-F(phi) - beta*G(phi).*ghat)
%                                   + tau*lambda*G(phi)
% and, since the update is LINEAR in lambda, the multiplier that restores
% the fluid volume sum(phi) = Vcells is obtained in closed form from two
% solves with the same (prefactorized) matrix ac.Adec.
%
% The box constraint 0 <= phi <= 1 is enforced TOGETHER with the volume:
% the update is phi(lambda) = min(max(pa + lambda*pb, 0), 1) and lambda is
% chosen so that sum(phi(lambda)) = Vcells. Because pb >= 0 (G >= 0 and the
% semi-implicit matrix is an M-matrix), sum(phi(lambda)) is nondecreasing in
% lambda; the closed-form multiplier of the unclipped update is exact whenever
% nothing is clipped, and otherwise lambda is found by bisection, so clipping
% no longer perturbs the volume. ac.vol_project = false restores the earlier
% behaviour (closed-form multiplier, then clipping).
%
% freeze: logical mask of design-frozen cells (inlet/outlet neighborhoods);
% phiFrozen: their fixed values, reimposed after each step.
% info (optional output), per inner step: vol_err = sum(phi) - Vcells after
% the update, phi_min/phi_max of the unclipped update, n_clip = number of
% entries the box constraint clipped, lambda = multiplier used.

info = struct('vol_err', zeros(ac.Kin,1), 'phi_min', zeros(ac.Kin,1), ...
              'phi_max', zeros(ac.Kin,1), 'n_clip', zeros(ac.Kin,1), 'lambda', zeros(ac.Kin,1));
for it = 1:ac.Kin
    Fw = phi.*(1-phi).*(0.5-phi);
    G  = phi.^2.*(1-phi).^2;
    if isfield(ac,'gate_norm') && ac.gate_norm
        % Normalize the GATED sensitivity. The design forcing only acts
        % through G'(phi), so the scale that matters is that of G.*sens;
        % normalizing sens globally throttles the forcing whenever the
        % sensitivity is dominated by regions the gate discards (as in the
        % eps^{-2}-penalty diffuse-domain formulation, where dJ/dphi is
        % concentrated in the fluid).
        gs = G.*sens;
        % Optional outlier clip (ac.clip_pct): the gated sensitivity is spiky at
        % geometric corners, and normalizing by a norm those few cells dominate
        % throttles the forcing everywhere else. Clip before normalizing.
        if isfield(ac,'clip_pct') && ac.clip_pct < 100
            act = abs(gs(G > 1e-9));
            if ~isempty(act)
                cap = prctile(act, ac.clip_pct);
                if cap > 0, gs = max(min(gs, cap), -cap); end
            end
        end
        forcing = -Fw - ac.beta*gs/(norm(gs(:)) + 1e-14);
    else
        ghat = sens / (norm(sens(:)) + 1e-14);
        forcing = -Fw - ac.beta*G.*ghat;
    end
    forcing(freeze) = 0;
    Gm = G; Gm(freeze) = 0;

    rhs0 = phi + ac.tau*forcing;
    rhs1 = ac.tau*Gm;
    if isfield(ac,'cvec')   % Dirichlet boundary contribution of the AC solve
        pa = ac.Adec \ (rhs0(:) + ac.cvec);
    else
        pa = ac.Adec \ rhs0(:);
    end
    pb = ac.Adec \ rhs1(:);
    pa = reshape(pa, size(phi)); pb = reshape(pb, size(phi));
    pa(freeze) = phiFrozen(freeze); pb(freeze) = 0;

    sb = sum(pb(:));
    if abs(sb) > 1e-12
        lambda = (Vcells - sum(pa(:))) / sb;      % exact for the unclipped update
        phiraw = pa + lambda*pb;
        if (~isfield(ac,'vol_project') || ac.vol_project) && (any(phiraw(:) < 0) || any(phiraw(:) > 1))
            lambda = project_vol(pa(:), pb(:), Vcells, lambda);
            phiraw = pa + lambda*pb;
        end
    else
        lambda = 0;      % no interface formed yet (binary phi): skip
        phiraw = pa;
    end
    phi = min(max(phiraw,0),1);
    phi(freeze) = phiFrozen(freeze);
    info.vol_err(it) = sum(phi(:)) - Vcells;
    info.phi_min(it) = min(phiraw(:)); info.phi_max(it) = max(phiraw(:));
    info.n_clip(it)  = nnz(phiraw < 0 | phiraw > 1);
    info.lambda(it)  = lambda;
end
end

% ------------------------------------------------------------------
function lambda = project_vol(pa, pb, Vcells, lam0)
% Bisection for f(lambda) = sum(min(max(pa + lambda*pb, 0), 1)) - Vcells = 0,
% f nondecreasing since pb >= 0. lam0 (the closed-form multiplier) seeds the
% bracket. If the volume is unreachable under the box constraint (f never
% changes sign), the closed-form multiplier is kept.
f = @(l) sum(min(max(pa + l*pb, 0), 1)) - Vcells;
tol = 1e-12*max(Vcells, 1);
f0 = f(lam0);
if abs(f0) <= tol, lambda = lam0; return, end
step = max(abs(lam0), 1);
if f0 < 0
    lo = lam0; hi = lam0 + step;
    while f(hi) < 0 && step < 1e30, step = 2*step; hi = lam0 + step; end
else
    hi = lam0; lo = lam0 - step;
    while f(lo) > 0 && step < 1e30, step = 2*step; lo = lam0 - step; end
end
if f(hi) < 0 || f(lo) > 0, lambda = lam0; return, end
for k = 1:200
    mid = 0.5*(lo + hi); fm = f(mid);
    if abs(fm) <= tol || (hi - lo) <= 1e-14*max(1, abs(mid)), lambda = mid; return, end
    if fm < 0, lo = mid; else, hi = mid; end
end
lambda = 0.5*(lo + hi);
end
