function report = test_cantilever_checks()
% TEST_CANTILEVER_CHECKS  Implementation checks of the cantilever model quoted
% in Section 4.4:
%   1. element patch test: rigid-body modes lie in the null space of K_e, and
%      a uniform strain gives the exact energy;
%   2. slender cantilever tip deflection against Euler-Bernoulli beam theory;
%   3. total sensitivity dJ/dphi_e (compliance plus f_V times volume) against
%      central finite differences;
%   4. reduced solve in a basis containing the full-order state;
%   5. design-update invariants: bounds, frozen cells, and an equilibrium
%      profile left almost unchanged when the forcing is switched off.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));

% ---- 1. patch test
par = struct('Nx', 4, 'Ny', 2, 'h', 0.25, 'E', 1, 'nu', 0.3, 'load', -1);
S = elast_base(par); Ke = S.Ke; h = par.h;
xy = [0 0; h 0; h h; 0 h];
rb = zeros(8, 3); rb(1:2:7, 1) = 1; rb(2:2:8, 2) = 1;
rb(1:2:7, 3) = -xy(:, 2); rb(2:2:8, 3) = xy(:, 1);
rigid_body = max(abs(Ke*rb), [], 'all');
ue = zeros(8, 1); ue(1:2:7) = 1e-3*xy(:, 1);
exact = 0.5*(par.E/(1 - par.nu^2))*(1e-3)^2*h^2;
uniform_strain = abs(0.5*(ue.'*Ke*ue) - exact)/exact;
assert(rigid_body < 1e-12 && uniform_strain < 1e-12, 'Element patch test failed');

% ---- 2. slender cantilever (unit thickness, nu = 0)
L = 2; Hb = 0.05; P = 1; nx = 160; hb = L/nx;
parb = struct('Nx', nx, 'Ny', round(Hb/hb), 'h', hb, 'E', 1, 'nu', 0, 'load', -P);
Sb = elast_base(parb);
K = elast_assemble(Sb, ones(parb.Nx*parb.Ny, 1));
U = zeros(Sb.ndof, 1); U(Sb.free) = K(Sb.free, Sb.free)\Sb.F(Sb.free);
I = (parb.Ny*hb)^3/12;
beam_ratio = U(2*Sb.nload)/(-P*L^3/(3*parb.E*I));
assert(abs(beam_ratio - 1) < 0.03, 'Beam deflection check failed');

% ---- 3. total sensitivity against central differences
p = cantilever_problem(struct('Nx', 20, 'Ny', 10, 'kmin', 1e-4));
rng(1); phi = 0.35 + 0.4*rand(20, 10);
o = p.fom(phi, [], false);
idx = [13, 57, 122, 199]; d = 1e-4; err = zeros(size(idx));   % J ~ 1e2: smaller steps hit round-off
for m = 1:numel(idx)
    pp = phi; pp(idx(m)) = pp(idx(m)) + d;
    pm = phi; pm(idx(m)) = pm(idx(m)) - d;
    fd = (p.fom(pp, [], false).J - p.fom(pm, [], false).J)/(2*d);
    err(m) = abs(fd - o.sens(idx(m)))/max(abs(fd), 1e-30);
end
sensitivity_fd = max(err);
assert(sensitivity_fd < 3e-5, 'Sensitivity finite-difference check failed');

% ---- 4. reduced solve with a basis containing the full-order state
rom = p.rom_new(struct('w_max', 8, 'r_max', 8, 'pod_tol', 1e-10, 'lsq', 'qr'));
rom = p.enrich(rom, o.U, [], phi);
r = p.rom_step(phi, rom, o.U);
rom_state_error = norm(r.U - o.U)/norm(o.U);
assert(rom_state_error < 1e-8, 'Reduced reproduction failed');

% ---- 5. design-update invariants
Lac = neumann_laplacian(20, 10, 0.1);
ac = struct('tau', 0.01, 'Kin', 5, 'beta', 5);
ac.Adec = decomposition(speye(200) - ac.tau*(4*0.1/(6*sqrt(2)))^2*Lac);
frz = false(size(phi)); frz(1, :) = true;
p1 = ac_update(phi, o.sens, frz, phi, ac);
bounds_ok = min(p1(:)) >= 0 && max(p1(:)) <= 1;
frozen_change = max(abs(p1(1, :) - phi(1, :)));
dist = ((1:20).' - 10.5)*0.1;
peq = repmat(0.5*(1 + tanh(3*dist/(4*0.1))), 1, 10);
p2 = ac_update(peq, zeros(size(peq)), false(size(peq)), peq, ac);
equilibrium_drift = max(abs(p2(:) - peq(:)));
assert(bounds_ok && frozen_change == 0 && equilibrium_drift < 1e-2, 'Design-update check failed');

report = table(rigid_body, uniform_strain, beam_ratio, sensitivity_fd, ...
    rom_state_error, frozen_change, equilibrium_drift);
disp(report);
fprintf('Cantilever implementation checks passed\n');
end
