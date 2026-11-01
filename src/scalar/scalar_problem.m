function prob = scalar_problem(opts)
% SCALAR_PROBLEM  Problem adapter for tr_optimize: the shape-identification
% problem of Section sec:num, in the diffuse-domain form
%   A(phi) T = b(phi),   A(phi) = eps^-2 I - diag(phi)(L_h + eps^-2 I),
%   b(phi) = phi.*f,     J = int (T - Tbar)^2,
% on Omega = (-0.15,1.25) x (-0.1,0.1) with T = 0 on the outer boundary; the
% target Tbar is the solution supported on the airfoil-like domain D_ref, and
% the initial design is a thin heptagon. The design is updated by the
% Allen-Cahn step (scalar_ac_update). gate_norm selects normalization of the
% gated sensitivity; false preserves the legacy before-gate normalization.
%
% Options: h (4e-3, nominal grid size), eps_fac (4: eps = eps_fac*h with the
% nominal h), tau (0.01), beta (1e3), Kin (10).
% Band-limited computation (Phase B of PLAN.md):
%   band (false): reduced step from data built once per basis update at a
%       reference design phi_b, corrected only on the band B = {|phi - phi_b|
%       > btol} (band_build, band_step): O(|B| r^2) plus three products of
%       length n (the reduced state, adjoint and the sensitivity norm the
%       ungated design forcing needs); the dual-weighted estimate is the
%       unenriched W_r' r_s; btol (1e-14); rho_max (0.3: rebuild at the current
%       design when |B| exceeds rho_max*n)
%   ac_band (false): narrow-band Allen-Cahn step on {phi(1-phi) > ac_tol}
%       dilated by ac_dil nodes, the design outside held fixed; ac_tol (1e-12),
%       ac_dil (2)
%
% Unknowns are the n = mx*my interior nodes. The adjoint is defined with the
% exact gradient of the discrete objective, A' W = dJ/dT, so that the
% dual-weighted estimate is J(T) - J(T_r) ~ W' r_s; the paper's adjoint
% 2(Tbar - T) differs from W by the factor -1/(hx*hy), which the normalized
% design update does not see.
def = struct('h', 4e-3, 'eps_fac', 4, 'tau', 0.01, 'beta', 1e3, 'Kin', 10, ...
    'band', false, 'btol', 1e-14, 'rho_max', 0.3, 'ac_band', false, 'ac_tol', 1e-12, 'ac_dil', 2, 'cache_ops', false, 'gate_norm', false);
if nargin < 1, opts = struct(); end
opts = set_defaults(opts, def);

% ---- grid (as problem_setting_shapeopt_eps4.m / _h002_eps4.m)
ax = -0.15; bx = 1.25; ay = -0.1; by = 0.1;
h0 = opts.h;
epsd = opts.eps_fac*h0;                       % eps uses the nominal h, as the setting files
c = epsd^(-2);
hat_eps = epsd/(6*sqrt(2));
mx = ceil((bx-ax)/h0) - 1; my = ceil((by-ay)/h0) - 1;
hx = (bx-ax)/(mx+1); hy = (by-ay)/(my+1);
x = linspace(ax, bx, mx+2); y = linspace(ay, by, my+2);
[X, Y] = ndgrid(x, y);
Iint = 2:mx+1; Jint = 2:my+1;
n = mx*my;
Tx = spdiags(ones(mx,1)*[1 -2 1], [-1 0 1], mx, mx);
Ty = spdiags(ones(my,1)*[1 -2 1], [-1 0 1], my, my);
L = kron(speye(my), Tx)/hx^2 + kron(Ty, speye(mx))/hy^2;
Lc = L + c*speye(n);

% ---- data
t = 0:.1:2;
xG = (t-1).^2; yG = 0.13*t.*(t-1).*(t-2);           % reference domain D_ref
u0 = @(x,y) (y.^2 - 0.0169*x.*(x-1).^2)*100;
Tbar_full = u0(X,Y).*double(inpolygon(X, Y, xG, yG));
f = (0.1014*X(Iint,Jint) - 2.0676)*100;  f = f(:);
Tbar = Tbar_full(Iint,Jint); Tbar = Tbar(:);
wq = hx*hy;                                          % quadrature weight of the objective

% ---- initial design: thin heptagon (binary), phi = 0 on the outer boundary
xB = [-0.05, 0.5, 0.75, 1.1, 0.75, 0.5, -0.05];
yB = [-0.06, -0.06, -0.05, 0, 0.05, 0.06, 0.06];
phi0 = double(inpolygon(X, Y, xB, yB));

% ---- Allen-Cahn operator
Adec = decomposition(speye(n) - opts.tau*hat_eps^2*L);

D = struct('mx', mx, 'my', my, 'n', n, 'hx', hx, 'hy', hy, 'c', c, 'eps', epsd, ...
    'L', L, 'Lc', Lc, 'f', f, 'Tbar', Tbar, 'wq', wq, 'Iint', Iint, 'Jint', Jint, ...
    'X', X, 'Y', Y, 'xG', xG, 'yG', yG, 'Adec', Adec, 'tau', opts.tau, 'beta', opts.beta, 'Kin', opts.Kin, ...
    'kappa', hat_eps^2, 'btol', opts.btol, 'rho_max', opts.rho_max, 'ac_tol', opts.ac_tol, 'ac_dil', opts.ac_dil, 'cache_ops', opts.cache_ops, 'gate_norm', opts.gate_norm);

prob = struct();
prob.name = sprintf('scalar h=%g', h0);
prob.phi0 = phi0;
prob.self_adjoint = false;
prob.fallback = 'precond';                   % GMRES on the retained LU: 4-8 iterations, as cheap as a new LU
prob.refresh_its = 8;                        % m_max: refactor once a state GMRES needs more
prob.aux_names = {};
prob.fom     = @(phi, Uwarm, adj_reuse) scalar_fom(D, phi);
prob.pc      = @(phi, U0, W0, Kref, opts) scalar_pc(D, phi, U0, W0, Kref, opts);
prob.rom_new = @(o) scalar_rom_new(D, o);
prob.correct = @(phi, rom, R, Kref, varargin) scalar_correct(D, phi, rom, R, Kref, varargin{:});
if opts.band
    prob.enrich   = @(rom, U, W, phib) band_build(D, scalar_rom_enrich(D, rom, U, W), phib);
    prob.rom_step = @(phi, rom, Uwarm) band_step(D, phi, rom);
else
    prob.enrich   = @(rom, U, W, varargin) scalar_rom_enrich(D, rom, U, W);
    prob.rom_step = @(phi, rom, Uwarm) scalar_rom_step(D, phi, rom);
end
if opts.ac_band
    prob.update = @(phi, sens) scalar_ac_update_band(D, phi, sens);
else
    prob.update = @(phi, sens) scalar_ac_update(D, phi, sens);
end
prob.res_s   = @(U, phi) scalar_res_s(D, U, phi);          % b - A(phi) U
prob.res_a   = @(W, U, phi) scalar_res_a(D, W, U, phi);    % dJ/dT(U) - A(phi)' W
prob.forcing = @(phi, sens) scalar_forcing(phi, sens, D.gate_norm);
prob.jac     = @(U, phi) assemble(D, phi);                 % A(phi), the state Jacobian
prob.info = struct('opts', opts, 'D', rmfield(D, {'Adec'}));
end

function r = scalar_res_s(D, T, phi)
[A, b] = assemble(D, phi);
r = b - A*T;
end

function r = scalar_res_a(D, W, T, phi)
A = assemble(D, phi);
[~, dJdT] = objective(D, T);
r = dJdT - A.'*W;
end

function f = scalar_forcing(phi, sens, gate_norm)
gs = phi.^2.*(1-phi).^2.*sens;
if gate_norm, den=norm(gs(:))+1e-14;
else, den=norm(sens(:))+1e-10;
end
f = gs/den;
end

% =====================================================================
function [A, b, phiv] = assemble(D, phi)
phiv = phi(D.Iint, D.Jint); phiv = phiv(:);
A = D.c*speye(D.n) - spdiags(phiv, 0, D.n, D.n)*D.Lc;
b = phiv.*D.f;
end

function [J, dJdT] = objective(D, T)
e = T - D.Tbar;
J = D.wq*(e.'*e);
dJdT = 2*D.wq*e;
end

function s = sens_full(D, phiv, T, W)
% dJ/dphi = W .* (L T + eps^-2 T + f) at the interior nodes, zero on the boundary
s = zeros(D.mx+2, D.my+2);
s(D.Iint, D.Jint) = reshape(W.*(D.Lc*T + D.f), D.mx, D.my);
end

function out = scalar_fom(D, phi)
% Full-order state and adjoint with one fresh LU factorization (A and A').
[A, b, phiv] = assemble(D, phi);
F = fact_lu(A);
T = fact_solve(F, A, b, false);
[J, dJdT] = objective(D, T);
W = fact_solve(F, A, dJdT, true);
out = struct('U', T, 'W', W, 'J', J, 'sens', sens_full(D, phiv, T, W), 'aux', zeros(1,0), 'Kref', F);
end

function [out, info] = scalar_pc(D, phi, U0, W0, Kref, opts)
% Full-order state and adjoint WITHOUT a new factorization: GMRES on A and A'
% preconditioned with the retained factorization K_ref (pc_gmres); refactor
% once the state solve needs more than opts.refresh_its iterations.
[A, b, phiv] = assemble(D, phi);
[T, fl, ~, it] = pc_gmres(A, b, U0, Kref.solve, opts.tol, opts.maxit);
info = struct('ok', fl == 0, 'its', [it 0], 'refreshed', false);
out = [];
if ~info.ok, return, end
[J, dJdT] = objective(D, T);
[W, flw, ~, itw] = pc_gmres(A.', dJdT, W0, Kref.solvet, opts.tol, opts.maxit);
info.its(2) = itw; info.ok = (flw == 0);
if ~info.ok, return, end
Fref = Kref;
if it > opts.refresh_its, Fref = fact_lu(A); info.refreshed = true; end
out = struct('U', T, 'W', W, 'J', J, 'sens', sens_full(D, phiv, T, W), 'aux', zeros(1,0), 'Kref', Fref);
end

function rom = scalar_rom_new(D, o)
rom = struct('SnapU', zeros(D.n,0), 'SnapW', zeros(D.n,0), 'Phi', zeros(D.n,0), 'PhiA', zeros(D.n,0), ...
    'LcPhi', zeros(D.n,0), 'r', 0, 'rA', 0, 'w_max', o.w_max, 'r_max', o.r_max, ...
    'pod_tol', o.pod_tol, 'lsq', o.lsq);
end

function rom = scalar_rom_enrich(D, rom, T, W)
% Separate state and adjoint windows; A(phi)*Psi_s = eps^-2 Psi_s - phi.*(Lc Psi_s)
% with Lc Psi_s formed once per basis update (the affine structure).
if ~isempty(T) || rom.r == 0                   % an adjoint-only correction leaves the state window as it is
    [rom.SnapU, rom.Phi, rom.r] = pod_window(rom.SnapU, T, rom.w_max, rom.r_max, rom.pod_tol);
    rom.LcPhi = D.Lc*rom.Phi;
end
[rom.SnapW, rom.PhiA, rom.rA] = pod_window(rom.SnapW, W, rom.w_max, rom.r_max, rom.pod_tol);
end

function R = scalar_rom_step(D, phi, rom)
% LSPG reduced state and adjoint, full-order residuals, dual-weighted estimate:
%   a = argmin || b - A Psi_s a ||,   T_r = Psi_s a,
%   b = argmin || dJ/dT - A' Psi_a b ||,   W_r = Psi_a b,
%   eta_J = w_+' r_s, w_+ the adjoint re-solved in span{Psi_a, r_s}.
phiv = phi(D.Iint, D.Jint); phiv = phiv(:);
b = phiv.*D.f;
APhi = D.c*rom.Phi - phiv.*rom.LcPhi;             % O(n r)
a = rom_lsq(APhi, b, rom.lsq);
T = rom.Phi*a;
rs = b - APhi*a;
[J, dJdT] = objective(D, T);
AtPhiA = D.c*rom.PhiA - D.Lc*(phiv.*rom.PhiA);    % A' = eps^-2 I - Lc diag(phi)
bA = rom_lsq(AtPhiA, dJdT, rom.lsq);
W = rom.PhiA*bA;
ra = dJdT - AtPhiA*bA;
[rdir, inspan] = dwr_direction(rs, rom.PhiA);
if inspan
    etaJ = W.'*rs;
else
    At2 = [AtPhiA, D.c*rdir - D.Lc*(phiv.*rdir)];
    b2 = rom_lsq(At2, dJdT, rom.lsq);
    etaJ = ([rom.PhiA, rdir]*b2).'*rs;
end
R = struct('U', T, 'W', W, 'J', J, 'sens', sens_full(D, phiv, T, W), 'aux', zeros(1,0), ...
    'etaE_s', norm(rs)/max(norm(b),1e-30), 'etaE_a', norm(ra)/max(norm(dJdT),1e-30), ...
    'rs', rs, 'ra', ra, 'dJdU', dJdT, 'etaJ', etaJ);
if D.cache_ops, R.ops=struct('phiv',phiv,'b',b,'AtPhiA',AtPhiA); end
end

function phi = scalar_ac_update(D, phi, sens)
% The Allen-Cahn step of the scalar problem (step4_AC.m of the original code,
% AC_scheme = 1): Kin semi-implicit steps with the sensitivity normalized by
% its Frobenius norm over the whole grid (ungated normalization),
%   (I - tau hat_eps^2 L) phi_new = phi + tau(-F'(phi) - beta G'(phi) dJ/||dJ||),
% on the interior nodes; phi stays 0 on the outer boundary.
nrm = norm(sens, 'fro') + 1e-10;
for i = 1:D.Kin
    Fw = phi.*(1-phi).*(0.5-phi);
    G  = phi.^2.*(1-phi).^2;
    if D.gate_norm, gs=G.*sens; nrm=norm(gs(:))+1e-14; end
    rhs = D.tau*(-Fw - D.beta*G.*sens/nrm) + phi;
    ri = rhs(D.Iint, D.Jint);
    phi(D.Iint, D.Jint) = reshape(D.Adec \ ri(:), D.mx, D.my);
end
end

function R = scalar_correct(D, phi, rom, R, Kref, mode)
% One retained-factor correction, with the adjoint evaluated at the new state.
% mode 'sa' (default): state and adjoint; 'a': adjoint only, the state, its
% residual and the objective of R kept (one triangular solve instead of two).
if nargin < 6 || isempty(mode), mode = 'sa'; end
if D.cache_ops
    if isfield(R,'ops'), ops=R.ops;
    else
        phiv=phi(D.Iint,D.Jint); phiv=phiv(:);
        ops=struct('phiv',phiv,'b',phiv.*D.f, ...
            'AtPhiA',D.c*rom.PhiA-D.Lc*(phiv.*rom.PhiA));
    end
    phiv=ops.phiv; b=ops.b;
else, [A, b, phiv] = assemble(D, phi); end
gm = isfield(rom, 'correction_method') && strcmp(rom.correction_method, 'gmres');   % GMRES on the residual through K_ref instead of one solve
[tol_s, tol_a] = corr_tols(rom, R);
Af  = @(v) D.c*v - phiv.*(D.Lc*v);           % A(phi) v   (the affine structure)
Atf = @(v) D.c*v - D.Lc*(phiv.*v);           % A(phi)' v
if strcmp(mode, 'a')
    U = R.U; J = R.J; g = R.dJdU; rs = R.rs;
    if gm, W = R.W + pc_gmres(Atf, R.ra, [], Kref.solvet, tol_a, rom.corr_maxit); else, W = R.W + Kref.solvet(R.ra); end
    ra = g - Atf(W);
else
if gm, U = R.U + pc_gmres(Af, R.rs, [], Kref.solve, tol_s, rom.corr_maxit); else, U = R.U + Kref.solve(R.rs); end
[J, g] = objective(D, U);
ra0 = g - Atf(R.W);
if gm, W = R.W + pc_gmres(Atf, ra0, [], Kref.solvet, tol_a, rom.corr_maxit); else, W = R.W + Kref.solvet(ra0); end
rs = b - Af(U); ra = g - Atf(W);
end
V = compact_dwr_basis(rom.PhiA, W, rs);
if D.cache_ops
    E=V(:,rom.rA+1:end); AtV=[ops.AtPhiA,D.c*E-D.Lc*(phiv.*E)];
else, AtV=A.'*V; end
z = rom_lsq(AtV, g, rom.lsq);
R = struct('U', U, 'W', W, 'J', J, 'sens', sens_full(D, phiv, U, W), ...
    'aux', zeros(1,0), 'etaE_s', norm(rs)/max(norm(b),1e-30), ...
    'etaE_a', norm(ra)/max(norm(g),1e-30), 'rs', rs, 'ra', ra, ...
    'dJdU', g, 'etaJ', (V*z).'*rs);
if D.cache_ops, R.ops=ops; end
end

% ===================== band-limited computation (Phase B) =====================
function rom = band_build(D, rom, phib)
% Data of the band-limited reduced step, formed once per basis update at the
% reference design phi_b (the design of the next reduced step), O(n r^2):
%   M  = [b(phi_b), A(phi_b) Psi_s] = Q R          (state residual r_s = M [1; -a] + band part)
%   Ma = [Psi_s, Tbar, A(phi_b)' Psi_a] = Qa Ra    (adjoint residual r_a = Ma [v; -b] + band part,
%                                                   v = [2w a; -2w], so that dJ/dT = Ma [v; 0])
% and Psi_a' M for the dual-weighted estimate.
phiv = phib(D.Iint, D.Jint); phiv = phiv(:);
M  = [phiv.*D.f, D.c*rom.Phi - phiv.*rom.LcPhi];
Ma = [rom.Phi, D.Tbar, D.c*rom.PhiA - D.Lc*(phiv.*rom.PhiA)];
[Q, Rm] = qr(M, 0);
[Qa, Ra] = qr(Ma, 0);
rom.band = struct('phiv', phiv, 'Q', Q, 'R', Rm, 'Qa', Qa, 'Ra', Ra, 'PaM', rom.PhiA.'*M);
end

function R = band_step(D, phi, rom)
% Band-limited reduced iteration. With delta = phi - phi_b supported on B,
%   r_s(a) = M [1; -a] + E_B Z [1; a],     Z = [delta_B.*f_B, delta_B.*(Lc Psi_s)_B],
%   r_a(b) = Ma [v; -b] + E_Bp Dl b,       Dl = Lc(Bp,B) (delta_B.*Psi_a(B,:)),
% Bp the rows Lc(:,B) reaches. Writing E Z = Q (Q_B' Z) + (I - Q Q') E Z,
%   ||r_s||^2 = ||(R S + Q_B' Z) [1; a]||^2 + ||L [1; a]||^2,  L'L = Z'Z - (Q_B'Z)'(Q_B'Z),
% S = diag(1, -I), so the LSPG state is a small stacked least-squares problem and
% the residual norm is a sum of squares (likewise the adjoint): O(|B| r^2).
% The reduced state and adjoint and the full sensitivity (whose Frobenius norm
% the ungated design forcing uses) cost three products of length n.
r = rom.r; rA = rom.rA; bd = rom.band; rebuilt = [];
phiv = phi(D.Iint, D.Jint); phiv = phiv(:);
dl = phiv - bd.phiv;
B = find(abs(dl) > D.btol);
if numel(B) > D.rho_max*D.n                    % the design has left the reference: rebuild
    rom = band_build(D, rom, phi); bd = rom.band; rebuilt = rom;
    dl = phiv - bd.phiv; B = find(abs(dl) > D.btol);
end
dB = dl(B);
% ---- state
Z = [dB.*D.f(B), dB.*rom.LcPhi(B,:)];          % |B| x (1+r)
Y = bd.Q(B,:).'*Z;
T1 = [bd.R*diag([1; -ones(r,1)]) + Y; psd_root(Z.'*Z - Y.'*Y)];
a = -(T1(:,2:end) \ T1(:,1));
rs = norm(T1*[1; a]); bn = norm(T1(:,1));      % ||r_s||, ||b(phi)||
% ---- adjoint
v = [2*D.wq*a; -2*D.wq];
[Bp, ~] = find(D.Lc(:,B)); Bp = unique(Bp);
Dl = D.Lc(Bp,B)*(dB.*rom.PhiA(B,:));           % |Bp| x rA
Ya = bd.Qa(Bp,:).'*Dl;
tv = bd.Ra(:,1:r+1)*v;                         % = Qa' dJ/dT
Ta = [-bd.Ra(:,r+2:end) + Ya; psd_root(Dl.'*Dl - Ya.'*Ya)];
ta = [tv; zeros(rA,1)];
bt = -(Ta \ ta);
ra = norm(ta + Ta*bt); gn = norm(tv);           % ||r_a||, ||dJ/dT||
J = gn^2/(4*D.wq);                              % w ||Psi a - Tbar||^2, as a sum of squares
% ---- unenriched dual-weighted estimate W_r' r_s = b' Psi_a' r_s
etaJ = bt.'*(bd.PaM*[1; -a] + rom.PhiA(B,:).'*(Z*[1; a]));
% ---- full-length reduced state, adjoint and sensitivity (three products of length n)
T = rom.Phi*a; W = rom.PhiA*bt;
s = zeros(D.mx+2, D.my+2);
s(D.Iint, D.Jint) = reshape(W.*(rom.LcPhi*a + D.f), D.mx, D.my);
R = struct('U', T, 'W', W, 'J', J, 'sens', s, 'aux', zeros(1,0), ...
    'etaE_s', rs/max(bn,1e-300), 'etaE_a', ra/max(gn,1e-300), 'rs', [], 'ra', [], 'dJdU', [], ...
    'etaJ', etaJ, 'res_fun', @() band_residuals(D, phi, T, W), 'rom', rebuilt, 'nB', numel(B));
end

function L = psd_root(K)
% L with L'L = K for a symmetric positive semidefinite K (small), robust to the
% rounding that can make K slightly indefinite
K = (K + K.')/2;
[V, E] = eig(K);
L = sqrt(max(diag(E), 0)).*V.';
end

function F = band_residuals(D, phi, T, W)
% the full-length residuals of a reduced pair, for calibration and error studies only
[A, b] = assemble(D, phi);
[~, dJdT] = objective(D, T);
F = struct('rs', b - A*T, 'ra', dJdT - A.'*W, 'dJdU', dJdT);
end

function phi = scalar_ac_update_band(D, phi, sens)
% Narrow-band version of scalar_ac_update: the Kin semi-implicit steps are
% solved only on the band {phi(1-phi) > ac_tol} dilated by ac_dil nodes, with
% the design outside it (0 or 1 to within ac_tol) held fixed as Dirichlet data.
% I - tau hat_eps^2 L is the identity up to tau hat_eps^2/h^2 ~ 2e-3, so each
% band system is solved by Jacobi sweeps (contraction ~1e-2 per sweep) to 1e-14.
nrm = norm(sens, 'fro') + 1e-10;
P = phi(D.Iint, D.Jint);
S = sens(D.Iint, D.Jint);
m = P.*(1-P) > D.ac_tol;
m = m | (conv2(double(P > 0.5), ones(3), 'same') > 0 & conv2(double(P <= 0.5), ones(3), 'same') > 0);  % sharp jumps
k = 2*D.ac_dil + 1;
m = conv2(double(m), ones(k), 'same') > 0;
in = find(m); out = find(~m);
if isempty(in), return, end
tk = D.tau*D.kappa;
Lbb = D.L(in,in);
dg = 1 - tk*full(diag(Lbb));                 % diagonal of the band operator
Off = tk*(Lbb - spdiags(full(diag(Lbb)), 0, numel(in), numel(in)));
cpl = tk*(D.L(in,out)*P(out));               % coupling to the frozen design outside
Pin = P(in); Sin = S(in);
% Include the frozen exterior in the global gated norm. Only its numerator
% stays fixed during this band solve; the norm is updated each substep.
gsout=P(out).^2.*(1-P(out)).^2.*S(out);
outside_norm=norm(gsout);
rho = max(sum(abs(Off), 2)./dg);             % Jacobi contraction (row-sum bound, ~1e-2)
nsweep = min(30, max(1, ceil(-14/log10(max(rho, 1e-30)))));   % sweeps to 1e-14, fixed in advance
for i = 1:D.Kin
    Fw = Pin.*(1-Pin).*(0.5-Pin);
    G  = Pin.^2.*(1-Pin).^2;
    if D.gate_norm, nrm=hypot(norm(G.*Sin),outside_norm)+1e-14; end
    rhs = Pin + D.tau*(-Fw - D.beta*G.*Sin/nrm) + cpl;
    x = rhs./dg;
    for sweep = 1:nsweep
        x = (rhs + Off*x)./dg;
    end
    Pin = x;
end
P(in) = Pin;
phi(D.Iint, D.Jint) = P;
end
