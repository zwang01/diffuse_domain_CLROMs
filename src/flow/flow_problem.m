function prob = flow_problem(opts)
% FLOW_PROBLEM  Problem adapter for tr_optimize: drag minimization for the
% isolated body of Gao and Ma (Test 2) at fixed solid volume, in the
% phi-weighted diffuse-domain formulation of Section sec:flows,
%   R(U;phi) = K(phi) U + N(U;phi) - F(phi) = 0,
%   K(phi) = K_bc + diag(phi_f) K_mom + diag(alpha*(1-phi_f)^q) + K_div diag(phi_f),
% on the MAC grid of (x0,xR) x (-1/2,1/2), h = 2/Nx, parabolic inflow
% u = 0.05 - 0.2 y^2, do-nothing outflow, nu = 0.02/Re. The objective is the
% momentum-balance drag (flow_drag_balance), whose discrete gradient is exact.
% The initial design is a disk of the prescribed volume with the equilibrium
% profile (balanced_ddm_problem sets relax_itr = 0 and builds it directly).
%
% The manuscript runs build this adapter through balanced_ddm_problem, which
% sets eps_fac = 4, tau = 0.01, Kin = 5 and the calibrated forcing coefficient
% (beta, the eta of the paper); the defaults listed here are not the published
% settings.
% run_submission_flow also passes penalty_C = 64, penalty_power = 2 and
% drag_form = 'traction'.
%
% Options: Nx (80: h = 0.025; 160: h = 0.0125), Re (40), eps_fac (4), beta
% (0.5), tau (0.2), Kin (5), clip_pct (100), Vsolid (0.1), x0 (-1), xR (3),
% relax_itr (250), vol_project (true), initial_aspect (1: the original disk;
% positive values give an area-preserving ellipse before relaxation).
%
% Full-order solves factor the saddle matrix with the low-fill LU of fact_lu
% (diagonal pivots accepted) and remove its error by iterative refinement.
def = struct('Nx', 80, 'Re', 40, 'eps_fac', 4, 'beta', 0.5, 'tau', 0.2, 'Kin', 5, ...
    'clip_pct', 100, 'Vsolid', 0.1, 'x0', -1, 'xR', 3, 'relax_itr', 250, 'vol_project', true, 'cache_ops', false, ...
    'initial_aspect', 1, 'conv_maps', true, 'penalty_C', NaN, 'penalty_power', 1, ...
    'drag_form', 'legacy');
if nargin < 1, opts = struct(); end
opts = set_defaults(opts, def);

par = struct();
par.h  = 2/opts.Nx;
par.Nx = round((opts.xR - opts.x0)/par.h); par.Ny = round(1/par.h);
par.x0 = opts.x0; par.y0 = -0.5;
par.nu = 0.02/opts.Re;
par.objective = 'drag_balance';
par.drag_form = opts.drag_form;
par.ubc = @(x,y) (abs(x - par.x0) < 1e-12).*(0.05 - 0.2*y.^2);
par.vbc = @(x,y) 0*x;
par.openRight = true(par.Ny,1); par.openBottom = false(par.Nx,1);
xc = par.x0 + ((1:par.Nx)-0.5)*par.h;
yc = par.y0 + ((1:par.Ny)-0.5)*par.h;
[Xc, Yc] = ndgrid(xc, yc);
eps_ac = opts.eps_fac*par.h; par.eps_ddm = eps_ac;
if isnan(opts.penalty_C), par.penalty_coeff=eps_ac^(-2);
else, par.penalty_coeff=opts.penalty_C*par.nu/eps_ac^2; end
par.penalty_power=opts.penalty_power;
hat_eps = eps_ac/(6*sqrt(2));                       % Allen-Cahn equilibrium = tanh(3 d/eps)

r0   = sqrt(opts.Vsolid/pi);
assert(isscalar(opts.initial_aspect) && isfinite(opts.initial_aspect) && opts.initial_aspect>0);
% Area-preserving ellipse before the usual volume projection and relaxation.
% Aspect 1 preserves the original disk initialization exactly.
phi  = 0.5*(1 + tanh(3*(sqrt(Xc.^2/opts.initial_aspect + Yc.^2*opts.initial_aspect) - r0)/eps_ac));
freeze = (Xc < par.x0+0.1) | (Xc > opts.xR-0.1) | (Yc < par.y0+0.1) | (Yc > par.y0+1-0.1);
phiFrozen = ones(par.Nx, par.Ny);
Vcells = ((opts.xR - opts.x0)*1 - opts.Vsolid)/par.h^2;

ac = struct('tau', opts.tau, 'Kin', opts.Kin, 'gate_norm', true, 'clip_pct', opts.clip_pct, ...
    'vol_project', opts.vol_project, 'beta', 0);
[Lac, bvec] = dirichlet_laplacian(par.Nx, par.Ny, par.h, ...
    ones(par.Ny,1), ones(par.Ny,1), ones(par.Nx,1), ones(par.Nx,1));
ac.Adec = decomposition(speye(par.Nx*par.Ny) - ac.tau*hat_eps^2*Lac);
ac.cvec = ac.tau*hat_eps^2*bvec;

% relax the interface to its equilibrium profile (no flow solves)
zs = zeros(par.Nx, par.Ny);
for k = 1:opts.relax_itr
    pn = ac_update_vol(phi, zs, freeze, phiFrozen, Vcells, ac);
    ad = max(abs(pn(:) - phi(:))); phi = pn;
    if ad < 5e-5, break, end
end
ac.beta = opts.beta;

S = ddm_stokes_base(par);
D = struct('par', par, 'S', S, 'ac', ac, 'freeze', freeze, 'phiFrozen', phiFrozen, 'Vcells', Vcells, 'cache_ops', opts.cache_ops, ...
           'conv_maps', opts.conv_maps, 'C', []);
if opts.conv_maps, D.C = flow_conv_maps(par, S); end     % stencil maps of the convection (reduced steps, corrections)

prob = struct();
prob.name = sprintf('flow h=%g', par.h);
prob.phi0 = phi;
prob.self_adjoint = false;
prob.fallback = 'precond';                  % Newton-GMRES on the retained LU: about half a fresh solve
prob.refresh_its = 40;                      % m_max over all Newton steps and the adjoint (one design
                                            % step from K_ref already takes ~20 in total)
prob.aux_names = {'Jvisc'};
prob.fom     = @(phi, Uwarm, adj_reuse) flow_fom(D, phi, Uwarm, adj_reuse);
prob.pc      = @(phi, U0, W0, Kref, o) flow_pc(D, phi, U0, W0, Kref, o);
prob.rom_new = @(o) flow_rom_new(D, o);
prob.correct = @(phi, rom, R, Kref, varargin) flow_correct(D, phi, rom, R, Kref, varargin{:});
prob.enrich  = @(rom, U, W, varargin) flow_rom_enrich(D, rom, U, W);
prob.rom_step = @(phi, rom, Uwarm) flow_rom_step(D, phi, rom, Uwarm);
prob.pair = @(phi, rom, U, W) flow_pair(D,phi,rom,U,W);
prob.update  = @(phi, sens) ac_update_vol(phi, sens, freeze, phiFrozen, Vcells, ac);
prob.res_s   = @(U, phi) flow_res_s(D, U, phi);            % F - K U - N(U)
prob.res_a   = @(W, U, phi) flow_res_a(D, W, U, phi);      % dJ/dU - (K + N'(U))' W
prob.forcing = @(phi, sens) flow_forcing(phi, sens, freeze);
prob.jac     = @(U, phi) flow_jac(D, U, phi);              % K(phi) + N'(U)
prob.sens_of = @(U, W, phi) flow_sens_of(D, U, W, phi);    % [sens, J, aux] at a given pair
prob.info = struct('opts', opts, 'par', par, 'xc', xc, 'yc', yc);
prob.err_mask = [true(S.Nu + S.Nv, 1); false(S.Np, 1)];   % error study: velocities (the pressure in the solid is set by the regularization)
if opts.conv_maps                           % handles for tests of the maps against ddm_nse_convection
    prob.conv = struct('KPsi', @(phi, rom) flow_KPsi(D, phi, rom), 'eval', @(cf, U) conv_eval(D, cf, U), ...
        'jac', @(cf, w, V) conv_jac_apply(D, cf, w, V), 'jact', @(cf, w, V) conv_jact_apply(D, cf, w, V), ...
        'phit', @(w, U, W) conv_phit_apply(D, w, U, W), 'K', @(cf, V) flow_K_apply(S, cf, V), 'Kt', @(cf, V) flow_Kt_apply(S, cf, V));
end
end

function r = flow_res_s(D, U, phi)
[K, F, phiu, phiv] = ddm_stokes_assemble(D.par, D.S, phi);
r = F - K*U - ddm_nse_convection(D.par, D.S, U, phiu, phiv);
end

function r = flow_res_a(D, W, U, phi)
[K, ~, phiu, phiv] = ddm_stokes_assemble(D.par, D.S, phi);
[~, dN] = ddm_nse_convection(D.par, D.S, U, phiu, phiv);
[~, dJdU] = flow_drag_balance(D.par, D.S, U);
r = dJdU - (K + dN).'*W;
end

function [sens, J, aux] = flow_sens_of(D, U, W, phi)
[~, ~, phiu, phiv] = ddm_stokes_assemble(D.par, D.S, phi);
[~, ~, dNdphif] = ddm_nse_convection(D.par, D.S, U, phiu, phiv);
[J, aux] = objective(D, U);
sens = sens_gather(D, U, W, dNdphif, phi);
end

function A = flow_jac(D, U, phi)
[K, ~, phiu, phiv] = ddm_stokes_assemble(D.par, D.S, phi);
[~, dN] = ddm_nse_convection(D.par, D.S, U, phiu, phiv);
A = K + dN;
end

function f = flow_forcing(phi, sens, freeze)
% the design term of ac_update_vol (gate_norm, no clipping): G'(phi) dJ, normalized after gating
gs = phi.^2.*(1-phi).^2.*sens;
f = gs/(norm(gs(:)) + 1e-14);
f(freeze) = 0;
end

% =====================================================================
function [J, Jvisc, dJdU] = objective(D, U)
par = D.par; S = D.S;
uv = U(1:S.Nu+S.Nv); g = S.E*uv + S.e0;
Jvisc = par.h^2*par.nu*(g.'*g);
[J, dJdU] = flow_drag_balance(par, S, U);
end

function sens = sens_gather(D, U, W, dNdphif, phi)
% dJ/dphi = -(dR/dphi)' W on the faces (momentum row scaling, penalty,
% weighted divergence, weighted convection), gathered to the cells. The drag
% has no explicit phi dependence.
par = D.par; S = D.S; Nx = par.Nx; Ny = par.Ny; Nu = S.Nu; Nv = S.Nv; Np = S.Np;
wmom  = [S.momu(:); S.momv(:); false(Np,1)];
phiu=zeros(Nx+1,Ny);phiu(2:Nx,:)=0.5*(phi(1:Nx-1,:)+phi(2:Nx,:));
phiu(1,:)=phi(1,:);phiu(Nx+1,:)=phi(Nx,:);
phiv=zeros(Nx,Ny+1);phiv(:,2:Ny)=0.5*(phi(:,1:Ny-1)+phi(:,2:Ny));
phiv(:,1)=phi(:,1);phiv(:,Ny+1)=phi(:,Ny);
phif=[phiu(:);phiv(:);ones(Np,1)];
dpen=-par.penalty_coeff*par.penalty_power*(1-phif).^(par.penalty_power-1);
dRmom = (S.Kmom*U - S.Fmom) + dpen.*U;
if issparse(dNdphif), cN = dNdphif.'*W; else, cN = dNdphif; end   % the convection term as a matrix, or as the vector (dN/dphi_f)' W
sface = wmom.*(W.*dRmom) + (S.Kdiv.'*W).*U + cN;
sfu = reshape(-sface(1:Nu), Nx+1, Ny);
sfv = reshape(-sface(Nu+1:Nu+Nv), Nx, Ny+1);
sens = zeros(Nx, Ny);
sens(1:Nx-1,:) = sens(1:Nx-1,:) + 0.5*sfu(2:Nx,:);
sens(2:Nx,:)   = sens(2:Nx,:)   + 0.5*sfu(2:Nx,:);
sens(1,:)      = sens(1,:)      + sfu(1,:);
sens(Nx,:)     = sens(Nx,:)     + sfu(Nx+1,:);
sens(:,1:Ny-1) = sens(:,1:Ny-1) + 0.5*sfv(:,2:Ny);
sens(:,2:Ny)   = sens(:,2:Ny)   + 0.5*sfv(:,2:Ny);
sens(:,1)      = sens(:,1)      + sfv(:,1);
sens(:,Ny)     = sens(:,Ny)     + sfv(:,Ny+1);
end

function [x, F] = lin_solve(F, A, b, transp)
% Solve with the factorization F (refinement); factor A afresh with the
% low-fill LU when F is empty or the refinement does not converge, and with
% UMFPACK's accurate pivoting if even that fails (never observed).
if ~isempty(F)
    [x, ok] = fact_solve(F, A, b, transp);
    if ok, return, end
end
F = fact_lu(A, [0.1 0]);
[x, ok] = fact_solve(F, A, b, transp);
if ok, return, end
F = fact_lu(A, [0.1 0.001]);
x = fact_solve(F, A, b, transp);
end

function out = flow_fom(D, phi, Uwarm, adj_reuse)
% Full-order Navier-Stokes solve with fresh factorizations: Newton with the
% exact Jacobian factored at every step (low-fill LU + refinement), the adjoint
% with the converged Jacobian factored afresh (adj_reuse = false, Algorithm 1)
% or with the last Newton factorization (adj_reuse = true, Algorithm 2).
par = D.par; S = D.S;
[K, F, phiu, phiv] = ddm_stokes_assemble(par, S, phi);
if isempty(Uwarm)
    U = lin_solve([], K, F, false);                       % Stokes start (one factorization)
else
    U = Uwarm;
end
nF = max(norm(F), 1e-14); Fj = [];
for it = 1:25
    [N, dN] = ddm_nse_convection(par, S, U, phiu, phiv);
    R = K*U + N - F; nr = norm(R)/nF;
    if nr < 1e-11, break, end
    [dU, Fj] = lin_solve([], K + dN, R, false);           % fresh factorization at every Newton step
    lam = 1;
    for bt = 1:8
        Ut = U - lam*dU;
        Nt = ddm_nse_convection(par, S, Ut, phiu, phiv);
        if norm(K*Ut + Nt - F)/nF < nr, break, end
        lam = lam/2;
    end
    U = U - lam*dU;
end
[N, dN, dNdphif] = ddm_nse_convection(par, S, U, phiu, phiv);
res = norm(K*U + N - F)/nF;
if res > 1e-8, warning('flow_fom: Newton residual %.2e', res); end
[J, Jvisc, dJdU] = objective(D, U);
if adj_reuse && ~isempty(Fj)
    [W, Fa] = lin_solve(Fj, K + dN, dJdU, true);
else
    [W, Fa] = lin_solve([], K + dN, dJdU, true);
end
out = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, dNdphif, phi), 'aux', Jvisc, 'Kref', Fa);
end

function [out, info] = flow_pc(D, phi, U0, W0, Kref, opts)
% Full-order solve WITHOUT a new factorization (fallback and verification of
% Algorithm 2): inexact Newton in which each Jacobian system is solved by
% GMRES preconditioned with the retained factorization K_ref, and the adjoint
% likewise with K_ref'. Refactor at the converged state once the solve needed
% more than opts.refresh_its GMRES iterations in total; info.ok = false when a
% GMRES call or Newton fails, and the caller then factors.
par = D.par; S = D.S;
[K, F, phiu, phiv] = ddm_stokes_assemble(par, S, phi);
nF = max(norm(F), 1e-14);
info = struct('ok', true, 'its', 0, 'refreshed', false); out = [];
U = U0;
for it = 1:25
    [N, dN] = ddm_nse_convection(par, S, U, phiu, phiv);
    R = K*U + N - F; nr = norm(R)/nF;
    if nr < 1e-11, break, end
    tol_k = max(opts.tol, min(1e-2, 0.1*nr));            % Eisenstat-Walker forcing
    [dU, fl, ~, git] = pc_gmres(K + dN, R, [], Kref.solve, tol_k, opts.maxit);
    info.its = info.its + git;
    if fl ~= 0, info.ok = false; return, end
    lam = 1;
    for bt = 1:6
        Ut = U - lam*dU;
        Nt = ddm_nse_convection(par, S, Ut, phiu, phiv);
        if norm(K*Ut + Nt - F)/nF < nr, break, end
        lam = lam/2;
    end
    U = U - lam*dU;
end
[N, dN, dNdphif] = ddm_nse_convection(par, S, U, phiu, phiv);
if norm(K*U + N - F)/nF > 1e-8, info.ok = false; return, end
[J, Jvisc, dJdU] = objective(D, U);
[W, flw, ~, gitw] = pc_gmres((K + dN).', dJdU, W0, Kref.solvet, opts.tol, opts.maxit);
info.its = info.its + gitw;
if flw ~= 0, info.ok = false; return, end
Fref = Kref;
if info.its > opts.refresh_its, Fref = fact_lu(K + dN, [0.1 0]); info.refreshed = true; end
out = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, dNdphif, phi), 'aux', Jvisc, 'Kref', Fref);
end

function rom = flow_rom_new(D, o)
Z = zeros(D.S.Ntot, 0);
rom = struct('SnapU', Z, 'SnapW', Z, 'Phi', Z, 'PhiA', Z, 'KbcPhi', Z, 'KmomPhi', Z, ...
    'KbctPhiA', Z, 'KdivtPhiA', Z, 'r', 0, 'rA', 0, 'w_max', o.w_max, 'r_max', o.r_max, ...
    'pod_tol', o.pod_tol, 'lsq', o.lsq, 'gn_maxit', 15);
if isfield(o, 'gn_maxit'), rom.gn_maxit = o.gn_maxit; end
if isfield(o, 'gn_stall'), rom.gn_stall = o.gn_stall; end
end

function rom = flow_rom_enrich(D, rom, U, W)
% Separate state and adjoint windows, and the design-independent products
% from which K(phi) Psi_s and K(phi)' Psi_a are formed in O(n r) (flow_KPsi).
S = D.S;
if ~isempty(U) || rom.r == 0                   % an adjoint-only correction leaves the state window as it is
    [rom.SnapU, rom.Phi, rom.r] = pod_window(rom.SnapU, U, rom.w_max, rom.r_max, rom.pod_tol);
    rom.KbcPhi    = S.Kbc*rom.Phi;
    rom.KmomPhi   = S.Kmom*rom.Phi;
    if D.conv_maps, rom.XP = D.C.X*rom.Phi; rom.YP = D.C.Y*rom.Phi; end   % advecting velocity of the basis (design independent)
end
[rom.SnapW, rom.PhiA, rom.rA] = pod_window(rom.SnapW, W, rom.w_max, rom.r_max, rom.pod_tol);
rom.KbctPhiA  = S.Kbc.'*rom.PhiA;
rom.KdivtPhiA = S.Kdiv.'*rom.PhiA;
end

function [KPhi, KtPhiA, cf] = flow_KPsi(D, phi, rom)
% K(phi) Psi_s and K(phi)' Psi_a from the precomputed products: elementwise
% factors times fixed operators (eq. rom_apply_flow), one sparse product each.
par = D.par; S = D.S; Nx = par.Nx; Ny = par.Ny; Np = S.Np;
if isfield(par,'delta_p'), dlt = par.delta_p; else, dlt = 1e-8/par.h; end
phiu = zeros(Nx+1,Ny); phiu(2:Nx,:) = 0.5*(phi(1:Nx-1,:) + phi(2:Nx,:));
phiu(1,:) = phi(1,:); phiu(Nx+1,:) = phi(Nx,:);
phiv = zeros(Nx,Ny+1); phiv(:,2:Ny) = 0.5*(phi(:,1:Ny-1) + phi(:,2:Ny));
phiv(:,1) = phi(:,1); phiv(:,Ny+1) = phi(:,Ny);
wmom = [S.momu(:); S.momv(:); false(Np,1)];
phif = [phiu(:); phiv(:); ones(Np,1)];
roww = wmom.*phif; pen = wmom.*par.penalty_coeff.*(1-phif).^par.penalty_power;
nug = [zeros(S.Nu+S.Nv,1); -dlt*ones(Np,1)];
KPhi   = rom.KbcPhi + roww.*rom.KmomPhi + pen.*rom.Phi + S.Kdiv*(phif.*rom.Phi) + nug.*rom.Phi;
KtPhiA = rom.KbctPhiA + S.Kmom.'*(roww.*rom.PhiA) + pen.*rom.PhiA + phif.*rom.KdivtPhiA + nug.*rom.PhiA;
cf = struct('phiu', phiu, 'phiv', phiv, 'roww', roww, 'pen', pen, 'phif', phif, 'nug', nug);
end

function R = flow_rom_step(D, phi, rom, U0)
% One reduced iteration: LSPG (Gauss-Newton) state warm-started from the last
% accepted state, LSPG adjoint, full-order residuals, dual-weighted estimate
% with the adjoint re-solved in span{Psi_a, r_s}, sensitivity.
if D.conv_maps, R = flow_rom_step_maps(D, phi, rom, U0); return, end
par = D.par; S = D.S;
[KPhi, KtPhiA, cf] = flow_KPsi(D, phi, rom);
F = S.Fbc + cf.roww.*S.Fmom;
Phi = rom.Phi;
a = Phi.'*U0; U = Phi*a;                    % warm start (Phi orthonormal)
nF = max(norm(F), 1e-14); eta = inf;
for it = 1:rom.gn_maxit
    [Nc, dN] = ddm_nse_convection(par, S, U, cf.phiu, cf.phiv);
    Rr = F - (KPhi*a + Nc);
    en = norm(Rr)/nF;
    if it > 1 && (en > eta || en < 1e-12), break, end
    eta = en;
    da = rom_lsq(KPhi + dN*Phi, Rr, rom.lsq);
    a = a + da; U = Phi*a;
    if norm(da) <= 1e-12*max(norm(a), 1), break, end
end
[Nc, dN, dNdphif] = ddm_nse_convection(par, S, U, cf.phiu, cf.phiv);   % at the final reduced state
rs = F - (KPhi*a + Nc);
[J, Jvisc, dJdU] = objective(D, U);
JtPhiA = KtPhiA + dN.'*rom.PhiA;
b = rom_lsq(JtPhiA, dJdU, rom.lsq);
W = rom.PhiA*b;
ra = dJdU - JtPhiA*b;
[rdir, inspan] = dwr_direction(rs, rom.PhiA);
if inspan
    etaJ = W.'*rs;
else
    Ktr = S.Kbc.'*rdir + S.Kmom.'*(cf.roww.*rdir) + cf.pen.*rdir + cf.phif.*(S.Kdiv.'*rdir) + cf.nug.*rdir;
    b2 = rom_lsq([JtPhiA, Ktr + dN.'*rdir], dJdU, rom.lsq);
    etaJ = ([rom.PhiA, rdir]*b2).'*rs;
end
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, dNdphif, phi), 'aux', Jvisc, ...
    'etaE_s', norm(rs)/nF, 'etaE_a', norm(ra)/max(norm(dJdU),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', dJdU, 'etaJ', etaJ);
if D.cache_ops, R.ops=struct('cf',cf,'F',F,'KtPhiA',KtPhiA); end
end

function R = flow_correct(D, phi, rom, R, Kref, mode)
% Correct state first, then the adjoint at that state; keep full residual checks.
% mode 'a' (adjoint only) is served by the stencil-map path.
if nargin < 6 || isempty(mode), mode = 'sa'; end
if D.conv_maps || strcmp(mode, 'a'), R = flow_correct_maps(D, phi, rom, R, Kref, mode); return, end
if D.cache_ops
    if isfield(R,'ops'), ops=R.ops;
    else
        [~,KtPhiA,cf]=flow_KPsi(D,phi,rom);
        ops=struct('cf',cf,'F',D.S.Fbc+cf.roww.*D.S.Fmom,'KtPhiA',KtPhiA);
    end
    F=ops.F; phiu=ops.cf.phiu; phiv=ops.cf.phiv;
    if ~isfield(ops,'K'), ops.K=ddm_stokes_assemble(D.par,D.S,phi); end
    K=ops.K;
else
    [K, F, phiu, phiv] = ddm_stokes_assemble(D.par, D.S, phi);
end
U = R.U + Kref.solve(R.rs);
[N, dN, dNdphif] = ddm_nse_convection(D.par, D.S, U, phiu, phiv);
[J, Jvisc, g] = objective(D, U);
A = K + dN;
W = R.W + Kref.solvet(g - A.'*R.W);
rs = F - K*U - N; ra = g - A.'*W;
V = compact_dwr_basis(rom.PhiA, W, rs);
if D.cache_ops
    AtV=[ops.KtPhiA,K.'*V(:,rom.rA+1:end)]+dN.'*V;
else, AtV=A.'*V; end
z = rom_lsq(AtV, g, rom.lsq);
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, dNdphif, phi), ...
    'aux', Jvisc, 'etaE_s', norm(rs)/max(norm(F),1e-14), ...
    'etaE_a', norm(ra)/max(norm(g),1e-30), 'rs', rs, 'ra', ra, ...
    'dJdU', g, 'etaJ', (V*z).'*rs);
if D.cache_ops, R.ops=ops; end
end

function R=flow_pair(D,phi,rom,U,W)
% Full residual and objective audit of externally computed reduced coordinates.
[K,F,pu,pv]=ddm_stokes_assemble(D.par,D.S,phi);
[N,dN,dphi]=ddm_nse_convection(D.par,D.S,U,pu,pv);
[J,Jvisc,g]=objective(D,U); A=K+dN;
rs=F-K*U-N; ra=g-A.'*W; V=compact_dwr_basis(rom.PhiA,W,rs);
z=rom_lsq(A.'*V,g,rom.lsq);
R=struct('U',U,'W',W,'J',J,'sens',sens_gather(D,U,W,dphi,phi),'aux',Jvisc, ...
    'etaE_s',norm(rs)/max(norm(F),1e-14),'etaE_a',norm(ra)/max(norm(g),1e-30), ...
    'rs',rs,'ra',ra,'dJdU',g,'etaJ',(V*z).'*rs);
if D.cache_ops
    [~,KtPhiA,cf]=flow_KPsi(D,phi,rom);
    R.ops=struct('K',K,'F',F,'KtPhiA',KtPhiA,'cf',cf);
end
end

% ====================================================================== stencil maps
function C = flow_conv_maps(par, S)
% FLOW_CONV_MAPS  Exact stencil factorization of the weighted convection of
% ddm_nse_convection. On the interior momentum rows C.rows, with q = phi_f.*U
% the weighted velocity and p = phi_f(C.rows),
%   N(U; phi_f) = (X U).*(Dx q + p.*gx) + (Y U).*(Dy q + p.*gy):
% X, Y interpolate the advecting velocity to the row's face (the face value
% for its own component, the four-point average for the other), Dx, Dy are
% the central differences of the convected quantity with the ghost
% conventions of ddm_nse_convection (wall ghost q = 2 phi_f u_bc - q_1, whose
% datum enters through gx, gy scaled by the face weight; open boundaries
% one-sided). With phi_f = 1 this is the factorization of eff_convection_maps
% of the Brinkman package. The maps are design independent, so a reduced
% iteration forms N, its Jacobian actions and the design derivative from a few
% sparse products of O(n) and dense products of O(n r), without assembly.
nx = par.Nx; ny = par.Ny; h = par.h; n = S.Ntot;
iu = @(i,j) i + (j-1)*(nx+1);
iv = @(i,j) S.Nu + i + (j-1)*nx;
[I, J] = ndgrid(2:nx, 1:ny); ru = iu(I, J); mu = numel(I); a = (1:mu)';
[P, Q] = ndgrid(1:nx, 2:ny); rv = iv(P, Q); mv = numel(P); b = mu + (1:mv)'; m = mu + mv;
C.rows = [ru(:); rv(:)];
cols = [ru(:); reshape(iu(P,Q-1),[],1); reshape(iu(P,Q),[],1); reshape(iu(P+1,Q-1),[],1); reshape(iu(P+1,Q),[],1)];
C.X = sparse([a; b; b; b; b], cols, [ones(mu,1); 0.25*ones(4*mv,1)], m, n);
cols = [reshape(iv(I-1,J),[],1); reshape(iv(I-1,J+1),[],1); reshape(iv(I,J),[],1); reshape(iv(I,J+1),[],1); rv(:)];
C.Y = sparse([a; a; a; a; b], cols, [0.25*ones(4*mu,1); ones(mv,1)], m, n);
% x derivatives: ordinary u stencil; v rows with left/right ghosts
openR = reshape(par.openRight(Q(:)-1) | par.openRight(Q(:)), size(Q));
vl = -ones(size(P))/(2*h); vr = -vl;
vl(P == 1) = 1/(2*h);                                   % wall ghost on the left
vr(P == nx & ~openR) = -1/(2*h);                        % wall ghost on the right; open boundary one-sided
C.Dx = sparse([a; a; b; b], ...
    [reshape(iu(I+1,J),[],1); reshape(iu(I-1,J),[],1); reshape(iv(max(P-1,1),Q),[],1); reshape(iv(min(P+1,nx),Q),[],1)], ...
    [ones(mu,1)/(2*h); -ones(mu,1)/(2*h); vl(:); vr(:)], m, n);
gx = zeros(size(P)); left = P == 1; right = P == nx & ~openR; yv = par.y0 + (Q-1)*h;
gx(left)  = -par.vbc(par.x0*ones(nnz(left),1), yv(left))/h;
gx(right) =  par.vbc((par.x0+nx*h)*ones(nnz(right),1), yv(right))/h;
C.gx = [zeros(mu,1); gx(:)];
% y derivatives: u rows with bottom/top ghosts; ordinary v stencil
openB = reshape(par.openBottom(I(:)-1) | par.openBottom(I(:)), size(I));
ud = -ones(size(I))/(2*h); uu = -ud;
ud(J == 1 & ~openB) = 1/(2*h); uu(J == ny) = -1/(2*h);
C.Dy = sparse([a; a; b; b], ...
    [reshape(iu(I,max(J-1,1)),[],1); reshape(iu(I,min(J+1,ny)),[],1); reshape(iv(P,Q+1),[],1); reshape(iv(P,Q-1),[],1)], ...
    [ud(:); uu(:); ones(mv,1)/(2*h); -ones(mv,1)/(2*h)], m, n);
gy = zeros(size(I)); bottom = J == 1 & ~openB; top = J == ny; xu = par.x0 + (I-1)*h;
gy(bottom) = -par.ubc(xu(bottom), par.y0*ones(nnz(bottom),1))/h;
gy(top)    =  par.ubc(xu(top), (par.y0+ny*h)*ones(nnz(top),1))/h;
C.gy = [gy(:); zeros(mv,1)];
end

function [N, w] = conv_eval(D, cf, U)
% The weighted convection N(U; phi_f) through the maps, and the row quantities
% the Jacobian actions reuse: w.x, w.y (advecting velocity), w.dx, w.dy
% (derivatives of the convected quantity, ghosts included).
C = D.C; p = cf.phif(C.rows); q = cf.phif.*U;
w.x = C.X*U; w.y = C.Y*U;
w.dx = C.Dx*q + p.*C.gx; w.dy = C.Dy*q + p.*C.gy;
N = zeros(D.S.Ntot, 1); N(C.rows) = w.x.*w.dx + w.y.*w.dy;
end

function Y = conv_jac_apply(D, cf, w, V, XV, YV)
% (dN/dU) V for the columns of V; XV = C.X*V, YV = C.Y*V may be passed when
% precomputed (the basis products rom.XP, rom.YP).
C = D.C;
if nargin < 5 || isempty(XV), XV = C.X*V; YV = C.Y*V; end
Q = cf.phif.*V;
Y = zeros(size(V));
Y(C.rows,:) = w.dx.*XV + w.x.*(C.Dx*Q) + w.dy.*YV + w.y.*(C.Dy*Q);
end

function Y = conv_jact_apply(D, cf, w, V)
% (dN/dU)' V for the columns of V
C = D.C; Vr = V(C.rows,:);
Y = C.X.'*(w.dx.*Vr) + cf.phif.*(C.Dx.'*(w.x.*Vr)) + C.Y.'*(w.dy.*Vr) + cf.phif.*(C.Dy.'*(w.y.*Vr));
end

function c = conv_phit_apply(D, w, U, W)
% (dN/dphi_f)' W: the convected quantity q = phi_f.*U and the wall ghosts
% 2 phi_f u_bc both depend on the face weight.
C = D.C; Wr = W(C.rows);
c = U.*(C.Dx.'*(w.x.*Wr) + C.Dy.'*(w.y.*Wr));
c(C.rows) = c(C.rows) + (C.gx.*w.x + C.gy.*w.y).*Wr;
end

function Y = flow_K_apply(S, cf, V)
% K(phi) V from the design-independent operators (the formula of flow_KPsi)
Y = S.Kbc*V + cf.roww.*(S.Kmom*V) + cf.pen.*V + S.Kdiv*(cf.phif.*V) + cf.nug.*V;
end

function Y = flow_Kt_apply(S, cf, V)
% K(phi)' V from the design-independent operators
Y = S.Kbc.'*V + S.Kmom.'*(cf.roww.*V) + cf.pen.*V + cf.phif.*(S.Kdiv.'*V) + cf.nug.*V;
end

function R = flow_rom_step_maps(D, phi, rom, U0)
% flow_rom_step through the stencil maps: no sparse assembly in the reduced
% iteration (the convection, its Jacobian actions on the bases and the design
% derivative come from conv_eval, conv_jac_apply, conv_jact_apply, conv_phit_apply).
S = D.S;
[KPhi, KtPhiA, cf] = flow_KPsi(D, phi, rom);
F = S.Fbc + cf.roww.*S.Fmom;
Phi = rom.Phi;
a = Phi.'*U0; U = Phi*a;                    % warm start (Phi orthonormal)
nF = max(norm(F), 1e-14); eta = inf;
if isfield(rom, 'gn_stall'), stall = rom.gn_stall; else, stall = 1; end   % stop once a step no longer reduces the residual by this factor
for it = 1:rom.gn_maxit
    [Nc, w] = conv_eval(D, cf, U);
    Rr = F - (KPhi*a + Nc);
    en = norm(Rr)/nF;
    if it > 1 && (en > stall*eta || en < 1e-12), break, end
    eta = en;
    da = rom_lsq(KPhi + conv_jac_apply(D, cf, w, Phi, rom.XP, rom.YP), Rr, rom.lsq);
    a = a + da; U = Phi*a;
    if norm(da) <= 1e-12*max(norm(a), 1), break, end
end
[Nc, w] = conv_eval(D, cf, U);              % at the final reduced state
rs = F - (KPhi*a + Nc);
[J, Jvisc, dJdU] = objective(D, U);
JtPhiA = KtPhiA + conv_jact_apply(D, cf, w, rom.PhiA);
b = rom_lsq(JtPhiA, dJdU, rom.lsq);
W = rom.PhiA*b;
ra = dJdU - JtPhiA*b;
[rdir, inspan] = dwr_direction(rs, rom.PhiA);
if inspan
    etaJ = W.'*rs;
else
    b2 = rom_lsq([JtPhiA, flow_Kt_apply(S, cf, rdir) + conv_jact_apply(D, cf, w, rdir)], dJdU, rom.lsq);
    etaJ = ([rom.PhiA, rdir]*b2).'*rs;
end
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, conv_phit_apply(D, w, U, W), phi), 'aux', Jvisc, ...
    'etaE_s', norm(rs)/nF, 'etaE_a', norm(ra)/max(norm(dJdU),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', dJdU, 'etaJ', etaJ, 'ops', struct('cf', cf, 'F', F, 'KtPhiA', KtPhiA));
end

function R = flow_correct_maps(D, phi, rom, R, Kref, mode)
% flow_correct through the stencil maps and the affine operators: no sparse
% assembly. The state is corrected first, then the adjoint at the corrected
% state; mode 'a' corrects the adjoint only (state, residual, objective kept).
if nargin < 6 || isempty(mode), mode = 'sa'; end
S = D.S;
if isfield(R, 'ops'), ops = R.ops; cf = ops.cf; F = ops.F; KtPhiA = ops.KtPhiA;
else, [~, KtPhiA, cf] = flow_KPsi(D, phi, rom); F = S.Fbc + cf.roww.*S.Fmom; end
gm = isfield(rom, 'correction_method') && strcmp(rom.correction_method, 'gmres');   % inexact Newton step: GMRES on the linearized residual through K_ref
[tol_s, tol_a] = corr_tols(rom, R);
if strcmp(mode, 'a')
    U = R.U; J = R.J; Jvisc = R.aux; g = R.dJdU; rs = R.rs;
    [~, w] = conv_eval(D, cf, U);
    if gm, W = R.W + pc_gmres(@(v) flow_Kt_apply(S, cf, v) + conv_jact_apply(D, cf, w, v), R.ra, [], Kref.solvet, tol_a, rom.corr_maxit);
    else, W = R.W + Kref.solvet(R.ra); end
else
if gm
    [~, w0] = conv_eval(D, cf, R.U);
    U = R.U + pc_gmres(@(v) flow_K_apply(S, cf, v) + conv_jac_apply(D, cf, w0, v), R.rs, [], Kref.solve, tol_s, rom.corr_maxit);
else
    U = R.U + Kref.solve(R.rs);
end
[N, w] = conv_eval(D, cf, U);
[J, Jvisc, g] = objective(D, U);
ra0 = g - flow_Kt_apply(S, cf, R.W) - conv_jact_apply(D, cf, w, R.W);
if gm, W = R.W + pc_gmres(@(v) flow_Kt_apply(S, cf, v) + conv_jact_apply(D, cf, w, v), ra0, [], Kref.solvet, tol_a, rom.corr_maxit);
else, W = R.W + Kref.solvet(ra0); end
rs = F - flow_K_apply(S, cf, U) - N;
end
ra = g - flow_Kt_apply(S, cf, W) - conv_jact_apply(D, cf, w, W);
V = compact_dwr_basis(rom.PhiA, W, rs);
E = V(:, rom.rA+1:end);
AtV = [KtPhiA + conv_jact_apply(D, cf, w, rom.PhiA), flow_Kt_apply(S, cf, E) + conv_jact_apply(D, cf, w, E)];
z = rom_lsq(AtV, g, rom.lsq);
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_gather(D, U, W, conv_phit_apply(D, w, U, W), phi), ...
    'aux', Jvisc, 'etaE_s', norm(rs)/max(norm(F),1e-14), 'etaE_a', norm(ra)/max(norm(g),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', g, 'etaJ', (V*z).'*rs, 'ops', struct('cf', cf, 'F', F, 'KtPhiA', KtPhiA));
end

