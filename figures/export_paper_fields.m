function export_paper_fields(outfile)
% EXPORT_PAPER_FIELDS  Collect the data of Figures 3-8 in one MATLAB v7 file
% (default results/figure_data.mat) for make_paper_figures.py. Designs and
% histories are read from the saved runs; the state fields of Figures 3 and 5
% are recomputed by full-order solves at the saved designs.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root, 'src')));
R = fullfile(root, 'results');
if nargin < 1, outfile = fullfile(R, 'figure_data.mat'); end

% ---- shape identification (Figures 3 and 4)
L = load(fullfile(R, 'common_balance_r03', 'scalar.mat'));
p = scalar_surface_problem(L.compact.info.opts); D = p.info.D;
observed = @(phi) phi.*embed(D, p.fom(phi, [], false).U);
s = histories(L);
s.x = D.X(:, 1); s.y = D.Y(1, :).';
s.observed = observed(L.compact.phi);
s.observed_init = observed(L.fom.phi0);
s.target = embed(D, D.Tbar);
s.domain = double(inpolygon(D.X, D.Y, D.xG, D.yG));
scalar = s; %#ok<NASGU>

% ---- drag minimization (Figures 5 and 6)
L = load(fullfile(R, 'flow_submission_traction', 'flow.mat'));
p = flow_problem(L.compact.info.opts);
par = p.info.par; S = ddm_stokes_base(par);
f = histories(L);
f.x = (par.x0 + (0:par.Nx)*par.h).'; f.y = (par.y0 + (0:par.Ny)*par.h).';
f.xc = p.info.xc(:); f.yc = p.info.yc(:);
f.psi = zeros(par.Nx+1, par.Ny+1, 2);
for k = 1:2
    if k == 1, phi = L.fom.phi0; else, phi = L.compact.phi; end
    o = p.fom(phi, [], false);
    u = reshape(o.U(1:S.Nu), par.Nx+1, par.Ny);
    [~, ~, phiu] = ddm_stokes_assemble(par, S, phi);
    f.psi(:, 2:end, k) = par.h*cumsum(phiu.*u, 2);
end
flow = f; %#ok<NASGU>

% ---- cantilever (Figures 7 and 8)
cant = histories(load(fullfile(R, 'common_balance_r03', 'cant_converged.mat'))); %#ok<NASGU>

save(outfile, 'scalar', 'flow', 'cant', '-v7');
fprintf('Saved %s\n', outfile);
end

function h = histories(L)
h = struct('phi0', L.fom.phi0, 'phi', L.compact.phi, 'J_fom', L.fom.J(:), ...
    'J_rom', L.compact.J(:), 'eta', L.compact.eta(:), 'tol', L.compact.tol(:), ...
    'source', L.compact.source(:));
end

function T = embed(D, v)
T = zeros(size(D.X)); T(D.Iint, D.Jint) = reshape(v, D.mx, D.my);
end
