function prob = cantilever_problem(opts)
% CANTILEVER_PROBLEM  Problem adapter for tr_optimize: the two-dimensional
% cantilever of Takezawa, Nishiwaki and Kitamura (2010, Section 5.1) in the
% diffuse-domain (ersatz-material) form of Section sec:elast,
%   K(phi) U = F,   K(phi) = sum_e k(phi_e) K_e,   k = kmin + (1-kmin) phi^p,
%   J = C + f*Vol,  C = F' U,  Vol = h^2 sum(phi),
% on a 2 x 1 Q1 grid, clamped on the left edge, unit downward load at the
% centre of the right edge. K(phi) is symmetric positive definite and the
% compliance is self-adjoint (the adjoint is the state), so the reduced model
% is a Galerkin (Ritz) projection on one basis and only the state indicator
% is screened.
%
% Options: Nx (200), Ny (100), p (3), kmin (1e-5), fvol (80), beta (5),
% eps_fac (4), tau (0.05), Kin (5), rad (0.12: hole radius of the
% perforated-plate initial design).
%
% The state vector of the engine is the vector of free degrees of freedom.
%
% The manuscript runs build this adapter through balanced_ddm_problem, which
% sets eps_fac = 4, tau = 0.01, Kin = 5 and the calibrated forcing coefficient
% (beta, the eta of the paper); the defaults listed here are not the published
% settings.
% The options bc = 'inverter' and objective = 'point' or 'target' are not used
% by the published runs.
def = struct('Nx', 200, 'Ny', 100, 'p', 3, 'kmin', 1e-5, 'fvol', 80, 'beta', 5, ...
    'eps_fac', 4, 'tau', 0.05, 'Kin', 5, 'rad', 0.12, 'cache_ops', false, 'element_ops', false, ...
    'objective', 'compliance', 'target_u', [], 'target_c', [], ...
    'bc', 'cantilever', 'k_in', 1, 'k_out', 1, 'volume', false, 'fallback', 'refactor');
% bc 'inverter': the compliant force inverter on a square domain (Nx = Ny):
% supports at the left corners (top and bottom tenth of the edge), unit
% horizontal input force at the left mid-edge node with a spring k_in, output
% node at the right mid-edge with a spring k_out; objective 'point' with
% J = u_x(out) (minimized: the output moves against the input) unless target_c
% is given. volume = true: exact volume constraint (ac_update_vol) at the
% initial volume instead of the volume penalty fvol.
% objective 'target': J = sum_i c_i (u_i - u0_i)^2 + fvol*Vol, the target-displacement
% objective of Section sec:elast (not self-adjoint: separate state and adjoint
% windows, Galerkin on both); target_u = u0 and target_c = c on the free dofs
% (or on all dofs, restricted here).
if nargin < 1, opts = struct(); end
opts = set_defaults(opts, def);

par = struct('Nx', opts.Nx, 'Ny', opts.Ny, 'h', 2/opts.Nx, 'E', 1, 'nu', 0.3, ...
    'p', opts.p, 'kmin', opts.kmin, 'load', -1);
S = elast_base(par);
Ksp = sparse(S.ndof, S.ndof);
if strcmp(opts.bc, 'inverter')
    Nx = par.Nx; Ny = par.Ny; nd = @(i, j) i + (j-1)*(Nx+1);
    js = find((0:Ny)*par.h <= 0.1*Ny*par.h + 1e-12 | (0:Ny)*par.h >= 0.9*Ny*par.h - 1e-12);   % corner supports on the left edge
    fixedNodes = nd(1, js(:));
    S.fixed = sort([2*fixedNodes-1; 2*fixedNodes]); S.free = setdiff((1:S.ndof).', S.fixed);
    jmid = round(Ny/2) + 1; nin = nd(1, jmid); nout = nd(Nx+1, jmid);
    F = zeros(S.ndof, 1); F(2*nin-1) = 1; S.F = F; S.nload = nin; S.nout = nout;
    Ksp(2*nin-1, 2*nin-1) = opts.k_in; Ksp(2*nout-1, 2*nout-1) = opts.k_out;
    if isempty(opts.target_c), opts.target_c = zeros(S.ndof, 1); opts.target_c(2*nout-1) = 1; end
    if strcmp(opts.objective, 'compliance'), opts.objective = 'point'; end
end
eps_ac = opts.eps_fac*par.h;
phi0 = cantilever_phi0(par, eps_ac, struct('rad', opts.rad));
hat_eps = eps_ac/(6*sqrt(2));
Lac = neumann_laplacian(par.Nx, par.Ny, par.h);
ac = struct('tau', opts.tau, 'Kin', opts.Kin, 'beta', opts.beta);
ac.Adec = decomposition(speye(par.Nx*par.Ny) - ac.tau*hat_eps^2*Lac);
freeze = false(par.Nx, par.Ny);

D = struct('par', par, 'S', S, 'Ff', S.F(S.free), 'Ksp', Ksp(S.free, S.free), 'fvol', opts.fvol, 'cache_ops', opts.cache_ops, 'element_ops', opts.element_ops);

prob = struct();
prob.name = sprintf('cantilever %dx%d', par.Nx, par.Ny);
prob.phi0 = phi0;
prob.self_adjoint = true;
% The current common-protocol study selects 'precond' explicitly to match
% its FOM comparator; the legacy adapter default preserves earlier studies.
prob.fallback = opts.fallback;
prob.refresh_its = 8;
prob.aux_names = {'C', 'Vol'};
prob.fom     = @(phi, Uwarm, adj_reuse) elast_fom(D, phi);
prob.pc      = @(phi, U0, W0, Kref, o) elast_pc(D, phi, U0, Kref, o);
prob.rom_new = @(o) struct('SnapU', zeros(numel(S.free),0), 'Phi', zeros(numel(S.free),0), ...
    'r', 0, 'rA', 0, 'w_max', o.w_max, 'r_max', o.r_max, 'pod_tol', o.pod_tol, 'lsq', o.lsq);
prob.enrich  = @(rom, U, W, varargin) elast_rom_enrich(D, rom, U);
prob.correct = @(phi, rom, R, Kref, varargin) elast_correct(D, phi, rom, R, Kref);   % self-adjoint: the state is the adjoint
prob.rom_step = @(phi, rom, Uwarm) elast_rom_step(D, phi, rom);
if opts.volume                                % exact volume constraint at the initial volume, gated normalization
    acv = ac; acv.gate_norm = true; acv.clip_pct = 100; acv.vol_project = true; Vcells = sum(phi0(:));
    prob.update = @(phi, sens) ac_update_vol(phi, sens, freeze, ones(par.Nx, par.Ny), Vcells, acv);
else
    prob.update  = @(phi, sens) ac_update(phi, sens, freeze, phi, ac);
end
prob.res_s   = @(U, phi) D.Ff - assemble(D, phi)*U;        % F - K(phi) U
prob.res_a   = [];                                         % self-adjoint: the adjoint is the state
prob.forcing = @(phi, sens) gated_forcing(phi, sens);
prob.jac     = @(U, phi) assemble(D, phi);                 % K(phi) on the free dofs
if ismember(opts.objective, {'target', 'point'})
    assert(~isempty(opts.target_c), 'the target/point objectives need target_c');
    assert(~opts.element_ops, 'the element-block operator is implemented for the compliance objective only');
    D.obj = opts.objective; D.cw = opts.target_c(:);
    if numel(D.cw) == S.ndof, D.cw = D.cw(S.free); end
    if strcmp(D.obj, 'target')
        assert(~isempty(opts.target_u), 'the target objective needs target_u');
        D.u0 = opts.target_u(:); if numel(D.u0) == S.ndof, D.u0 = D.u0(S.free); end
        assert(numel(D.u0) == numel(S.free));
    else
        D.u0 = [];
    end
    assert(numel(D.cw) == numel(S.free));
    prob.name = sprintf('cantilever %dx%d (%s displacement)', par.Nx, par.Ny, D.obj);
    prob.self_adjoint = false;
    prob.aux_names = {'Jm', 'Vol'};
    prob.fom      = @(phi, Uwarm, adj_reuse) elast_fom_t(D, phi);
    prob.pc       = @(phi, U0, W0, Kref, o) elast_pc_t(D, phi, U0, W0, Kref, o);
    prob.rom_new  = @(o) elast_rom_new_t(D, o);
    prob.enrich   = @(rom, U, W, varargin) elast_rom_enrich_t(D, rom, U, W);
    prob.correct  = @(phi, rom, R, Kref, varargin) elast_correct_t(D, phi, rom, R, Kref, varargin{:});
    prob.rom_step = @(phi, rom, Uwarm) elast_rom_step_t(D, phi, rom);
    prob.res_a    = @(W, U, phi) res_a_t(D, W, U, phi);
end
prob.info = struct('opts', opts, 'par', par);
end

function f = gated_forcing(phi, sens)
% the design term of ac_update: G'(phi) dJ, normalized after gating
gs = phi.^2.*(1-phi).^2.*sens;
f = gs/(norm(gs(:)) + 1e-14);
end

% =====================================================================
function [Kff, kphi] = assemble(D, phi)
par = D.par;
kphi = par.kmin + (1 - par.kmin)*phi(:).^par.p;
K = elast_assemble(D.S, kphi);
Kff = K(D.S.free, D.S.free) + D.Ksp;          % springs of the inverter (empty otherwise)
end

function [J, sens, aux] = objective(D, phi, Uf)
% J = C + fvol*Vol and its total sensitivity dJ/dphi_e = -k'(phi_e) u_e'K_e u_e
% + fvol*h^2 (the adjoint of the compliance is the state).
par = D.par; S = D.S;
U = zeros(S.ndof,1); U(S.free) = Uf;
C = D.Ff.'*Uf;
Vol = sum(phi(:))*par.h^2;
J = C + D.fvol*Vol;
ue = U(S.edof);
ce = sum((ue*S.Ke).*ue, 2);
dk = par.p*(1 - par.kmin)*phi(:).^(par.p - 1);
sens = reshape(-dk.*ce, size(phi)) + D.fvol*par.h^2;
aux = [C, Vol];
end

function out = elast_fom(D, phi)
% Full-order solve with a fresh AMD-ordered Cholesky factorization.
Kff = assemble(D, phi);
F = fact_chol(Kff);
Uf = F.solve(D.Ff);
[J, sens, aux] = objective(D, phi, Uf);
out = struct('U', Uf, 'W', [], 'J', J, 'sens', sens, 'aux', aux, 'Kref', F);
end

function [out, info] = elast_pc(D, phi, U0, Kref, opts)
% Full-order solve WITHOUT a new factorization: conjugate gradients
% preconditioned with the retained Cholesky factorization K_ref; refactor once
% the solve needs more than opts.refresh_its iterations.
Kff = assemble(D, phi);
[Uf, fl, ~, it] = pc_cg(Kff, D.Ff, U0, Kref.solve, opts.tol, opts.maxit);
info = struct('ok', fl == 0, 'its', it, 'refreshed', false); out = [];
if ~info.ok, return, end
Fref = Kref;
if it > opts.refresh_its, Fref = fact_chol(Kff); info.refreshed = true; end
[J, sens, aux] = objective(D, phi, Uf);
out = struct('U', Uf, 'W', [], 'J', J, 'sens', sens, 'aux', aux, 'Kref', Fref);
end

function rom = elast_rom_enrich(D, rom, Uf)
[rom.SnapU, rom.Phi, rom.r] = pod_window(rom.SnapU, Uf, rom.w_max, rom.r_max, rom.pod_tol);
rom.rA = rom.r;
if D.element_ops
    r=rom.r; U=zeros(D.S.ndof,r); U(D.S.free,:)=rom.Phi;
    E=cell(r,1); KE=cell(r,1);
    for j=1:r, E{j}=reshape(U(D.S.edof(:),j),size(D.S.edof)); KE{j}=E{j}*D.S.Ke; end
    ids=find(triu(true(r))); C=zeros(size(D.S.edof,1),numel(ids));
    for k=1:numel(ids), [i,j]=ind2sub([r,r],ids(k)); C(:,k)=sum(E{i}.*KE{j},2); end
    rom.element=struct('C',C,'ids',ids,'rhs',rom.Phi.'*D.Ff);
end
end

function R = elast_rom_step(D, phi, rom)
if D.element_ops, R=element_step(D,phi,rom); return, end
% Galerkin (Ritz) reduced state, its full-order residual, and the
% dual-weighted estimate of the compliance error: the adjoint (= the state)
% re-solved by Galerkin in span{Psi, r_s}, eta_J = w_+' r_s, which estimates
% C - C_r = e' K e >= 0.
Kff = assemble(D, phi);
Phi = rom.Phi;
KP = Kff*Phi;
Kr = Phi.'*KP; Kr = (Kr + Kr.')/2;
a = Kr \ (Phi.'*D.Ff);
Uf = Phi*a;
rs = D.Ff - KP*a;
[J, sens, aux] = objective(D, phi, Uf);
[rdir, inspan] = dwr_direction(rs, Phi);
if inspan
    etaJ = 0;
else
    V = [Phi, rdir]; KV = [KP, Kff*rdir];
    M = V.'*KV; M = (M + M.')/2;
    w = V*(M \ (V.'*D.Ff));
    etaJ = w.'*rs;
end
R = struct('U', Uf, 'W', [], 'J', J, 'sens', sens, 'aux', aux, ...
    'etaE_s', norm(rs)/max(norm(D.Ff),1e-30), 'etaE_a', 0, ...
    'rs', rs, 'ra', [], 'dJdU', [], 'etaJ', etaJ);
if D.cache_ops, R.ops = struct('K',Kff,'KP',KP); end
end

function R = elast_correct(D, phi, rom, R, Kref)
if D.element_ops, R=element_correct(D,phi,rom,R,Kref); return, end
% The compliance adjoint equals the state; one triangular-solve pair suffices.
if D.cache_ops && isfield(R,'ops')
    ops = R.ops; K = ops.K;
else
    K = assemble(D, phi); ops = struct('K',K);
    if D.cache_ops, ops.KP = K*rom.Phi; end
end
if isfield(rom,'correction_method') && strcmp(rom.correction_method,'gmres')   % CG on K(phi) through the retained Cholesky factor
    tol_s = corr_tols(rom, R);
    U = R.U + pc_cg(K, R.rs, [], Kref.solve, tol_s, rom.corr_maxit);
else
z = Kref.solve(R.rs);
if isfield(rom,'correction_method') && strcmp(rom.correction_method,'ritz')
    Vc = compact_dwr_basis(rom.Phi, R.U, z);
    Mc = Vc.'*(K*Vc); Mc = (Mc+Mc.')/2;
    U = Vc*(Mc\(Vc.'*D.Ff));
else
    U = R.U + z;
end
end
rs = D.Ff - K*U;
[J, sens, aux] = objective(D, phi, U);
V = compact_dwr_basis(rom.Phi, U, rs);
if D.cache_ops, KV = [ops.KP, K*V(:,rom.r+1:end)]; else, KV = K*V; end
M = V.'*KV; M = (M + M.')/2;
z = M \ (V.'*D.Ff);
R = struct('U', U, 'W', [], 'J', J, 'sens', sens, 'aux', aux, ...
    'etaE_s', norm(rs)/max(norm(D.Ff),1e-30), 'etaE_a', 0, ...
    'rs', rs, 'ra', [], 'dJdU', [], 'etaJ', (V*z).'*rs);
if D.cache_ops, R.ops = ops; end
end

function [Kr,kp]=element_matrix(D,phi,rom)
kp=D.par.kmin+(1-D.par.kmin)*phi(:).^D.par.p;
Kr=zeros(rom.r); Kr(rom.element.ids)=rom.element.C.'*kp;
Kr=Kr+triu(Kr,1).';
end
function Y=element_apply(D,kp,X)
Y=zeros(size(X));
for j=1:size(X,2)
    u=zeros(D.S.ndof,1); u(D.S.free)=X(:,j);
    ue=u(D.S.edof); fe=kp.*(ue*D.S.Ke);
    y=accumarray(D.S.edof(:),fe(:),[D.S.ndof,1]); Y(:,j)=y(D.S.free);
end
end
function eta=element_dwr(D,rom,Kr,kp,U,rs)
V=compact_dwr_basis(rom.Phi,U,rs); E=V(:,rom.r+1:end);
KE=element_apply(D,kp,E); cross=rom.Phi.'*KE;
M=[Kr,cross;cross.',E.'*KE]; M=(M+M.')/2;
z=M\(V.'*D.Ff); eta=(V*z).'*rs;
end
function R=element_step(D,phi,rom)
[Kr,kp]=element_matrix(D,phi,rom); a=Kr\rom.element.rhs; U=rom.Phi*a;
rs=D.Ff-element_apply(D,kp,U); [J,sens,aux]=objective(D,phi,U);
R=struct('U',U,'W',[],'J',J,'sens',sens,'aux',aux, ...
    'etaE_s',norm(rs)/max(norm(D.Ff),1e-30),'etaE_a',0,'rs',rs,'ra',[], ...
    'dJdU',[],'etaJ',element_dwr(D,rom,Kr,kp,U,rs),'ops',struct('Kr',Kr,'kp',kp));
end
function R=element_correct(D,phi,rom,R,F)
assert(~isfield(rom,'correction_method') || ismember(rom.correction_method,{'defect','gmres'}), ...
    'Element operator experiment supports the defect and the CG correction.');
if isfield(R,'ops') && isfield(R.ops,'kp'), Kr=R.ops.Kr; kp=R.ops.kp;
else, [Kr,kp]=element_matrix(D,phi,rom); end
if isfield(rom,'correction_method') && strcmp(rom.correction_method,'gmres')   % CG on K(phi) through the retained Cholesky factor
    tol_s = corr_tols(rom, R);
    U=R.U+pc_cg(@(v) element_apply(D,kp,v), R.rs, [], F.solve, tol_s, rom.corr_maxit);
else
    U=R.U+F.solve(R.rs);
end
rs=D.Ff-element_apply(D,kp,U);
[J,sens,aux]=objective(D,phi,U);
R=struct('U',U,'W',[],'J',J,'sens',sens,'aux',aux, ...
    'etaE_s',norm(rs)/max(norm(D.Ff),1e-30),'etaE_a',0,'rs',rs,'ra',[], ...
    'dJdU',[],'etaJ',element_dwr(D,rom,Kr,kp,U,rs),'ops',struct('Kr',Kr,'kp',kp));
end

% ===================== target-displacement objective (not self-adjoint) =====================
function [J, g, aux] = objective_t(D, phi, Uf)
% 'target': J = sum_i c_i (u_i - u0_i)^2 + fvol*Vol on the free dofs, g = dJ/dU;
% 'point' : J = sum_i c_i u_i + fvol*Vol (a displacement functional, adjoint load c)
Vol = sum(phi(:))*D.par.h^2;
if strcmp(D.obj, 'point')
    Jm = sum(D.cw.*Uf); g = D.cw;
else
    e = Uf - D.u0;
    Jm = sum(D.cw.*e.^2); g = 2*D.cw.*e;
end
J = Jm + D.fvol*Vol;
aux = [Jm, Vol];
end

function sens = sens_t(D, phi, Uf, Wf)
% dJ/dphi_e = -k'(phi_e) u_e' K_e w_e + fvol h^2, W the adjoint K W = dJ/dU
par = D.par; S = D.S;
U = zeros(S.ndof,1); U(S.free) = Uf; W = zeros(S.ndof,1); W(S.free) = Wf;
ue = U(S.edof); we = W(S.edof);
ce = sum((ue*S.Ke).*we, 2);
dk = par.p*(1 - par.kmin)*phi(:).^(par.p - 1);
sens = reshape(-dk.*ce, size(phi)) + D.fvol*par.h^2;
end

function r = res_a_t(D, W, U, phi)
K = assemble(D, phi); [~, g] = objective_t(D, phi, U);
r = g - K*W;
end

function out = elast_fom_t(D, phi)
Kff = assemble(D, phi); F = fact_chol(Kff);
Uf = F.solve(D.Ff);
[J, g, aux] = objective_t(D, phi, Uf);
Wf = F.solve(g);
out = struct('U', Uf, 'W', Wf, 'J', J, 'sens', sens_t(D, phi, Uf, Wf), 'aux', aux, 'Kref', F);
end

function [out, info] = elast_pc_t(D, phi, U0, W0, Kref, opts)
% state and adjoint by conjugate gradients on the retained Cholesky factor
Kff = assemble(D, phi);
[Uf, fl, ~, it] = pc_cg(Kff, D.Ff, U0, Kref.solve, opts.tol, opts.maxit);
info = struct('ok', fl == 0, 'its', [it 0], 'refreshed', false); out = [];
if ~info.ok, return, end
[J, g, aux] = objective_t(D, phi, Uf);
[Wf, flw, ~, itw] = pc_cg(Kff, g, W0, Kref.solve, opts.tol, opts.maxit);
info.its(2) = itw; info.ok = flw == 0;
if ~info.ok, return, end
Fref = Kref;
if it > opts.refresh_its, Fref = fact_chol(Kff); info.refreshed = true; end
out = struct('U', Uf, 'W', Wf, 'J', J, 'sens', sens_t(D, phi, Uf, Wf), 'aux', aux, 'Kref', Fref);
end

function rom = elast_rom_new_t(D, o)
nf = numel(D.S.free);
rom = struct('SnapU', zeros(nf,0), 'Phi', zeros(nf,0), 'SnapW', zeros(nf,0), 'PhiA', zeros(nf,0), ...
    'r', 0, 'rA', 0, 'w_max', o.w_max, 'r_max', o.r_max, 'pod_tol', o.pod_tol, 'lsq', o.lsq);
end

function rom = elast_rom_enrich_t(D, rom, U, W) %#ok<INUSL>
if ~isempty(U) || rom.r == 0
    [rom.SnapU, rom.Phi, rom.r] = pod_window(rom.SnapU, U, rom.w_max, rom.r_max, rom.pod_tol);
end
[rom.SnapW, rom.PhiA, rom.rA] = pod_window(rom.SnapW, W, rom.w_max, rom.r_max, rom.pod_tol);
end

function R = elast_rom_step_t(D, phi, rom)
% Galerkin state and adjoint on separate bases (K symmetric positive definite),
% their full-order residuals, and the dual-weighted estimate with the adjoint
% re-solved in span{Psi_a, r_s}.
K = assemble(D, phi);
Phi = rom.Phi; PhiA = rom.PhiA;
KP = K*Phi; Kr = Phi.'*KP; Kr = (Kr + Kr.')/2;
a = Kr \ (Phi.'*D.Ff); U = Phi*a; rs = D.Ff - KP*a;
[J, g, aux] = objective_t(D, phi, U);
KPA = K*PhiA; KrA = PhiA.'*KPA; KrA = (KrA + KrA.')/2;
b = KrA \ (PhiA.'*g); W = PhiA*b; ra = g - KPA*b;
[rdir, inspan] = dwr_direction(rs, PhiA);
if inspan
    etaJ = W.'*rs;
else
    V = [PhiA, rdir]; KV = [KPA, K*rdir]; M = V.'*KV; M = (M + M.')/2;
    etaJ = (V*(M \ (V.'*g))).'*rs;
end
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_t(D, phi, U, W), 'aux', aux, ...
    'etaE_s', norm(rs)/max(norm(D.Ff),1e-30), 'etaE_a', norm(ra)/max(norm(g),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', g, 'etaJ', etaJ, 'ops', struct('K', K, 'KPA', KPA));
end

function R = elast_correct_t(D, phi, rom, R, Kref, mode)
% retained-factor correction of state and adjoint ('sa') or adjoint only ('a');
% 'gmres' correction method = conjugate gradients on K(phi) through K_ref
if nargin < 6 || isempty(mode), mode = 'sa'; end
if isfield(R, 'ops'), K = R.ops.K; KPA = R.ops.KPA; else, K = assemble(D, phi); KPA = K*rom.PhiA; end
gm = isfield(rom, 'correction_method') && strcmp(rom.correction_method, 'gmres');
[tol_s, tol_a] = corr_tols(rom, R);
if strcmp(mode, 'a')
    U = R.U; J = R.J; g = R.dJdU; aux = R.aux; rs = R.rs;
    if gm, W = R.W + pc_cg(K, R.ra, [], Kref.solve, tol_a, rom.corr_maxit); else, W = R.W + Kref.solve(R.ra); end
else
    if gm, U = R.U + pc_cg(K, R.rs, [], Kref.solve, tol_s, rom.corr_maxit); else, U = R.U + Kref.solve(R.rs); end
    [J, g, aux] = objective_t(D, phi, U);
    ra0 = g - K*R.W;
    if gm, W = R.W + pc_cg(K, ra0, [], Kref.solve, tol_a, rom.corr_maxit); else, W = R.W + Kref.solve(ra0); end
    rs = D.Ff - K*U;
end
ra = g - K*W;
V = compact_dwr_basis(rom.PhiA, W, rs); E = V(:, rom.rA+1:end);
KV = [KPA, K*E]; M = V.'*KV; M = (M + M.')/2;
etaJ = (V*(M \ (V.'*g))).'*rs;
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_t(D, phi, U, W), 'aux', aux, ...
    'etaE_s', norm(rs)/max(norm(D.Ff),1e-30), 'etaE_a', norm(ra)/max(norm(g),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', g, 'etaJ', etaJ, 'ops', struct('K', K, 'KPA', KPA));
end
