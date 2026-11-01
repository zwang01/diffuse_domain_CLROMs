function N = export_paper_numbers(outfile)
% EXPORT_PAPER_NUMBERS  Recompute every number reported in the numerical
% section of the manuscript from the saved results, print Tables 3-8, and
% write them to results/paper_numbers.json (or OUTFILE).
%
% Compare the scientific quantities with reference/published_numbers.json.
% Wall times and speedups depend on the workstation.
root = fileparts(fileparts(mfilename('fullpath')));
R = fullfile(root, 'results');
if nargin < 1, outfile = fullfile(R, 'paper_numbers.json'); end
cb = fullfile(R, 'common_balance_r03');
fl = fullfile(R, 'flow_submission_traction');

S  = load(fullfile(cb, 'scalar.mat'));
F  = load(fullfile(fl, 'flow.mat'));
C  = load(fullfile(cb, 'cant_converged.mat'));
S5 = load(fullfile(cb, 'scalar_tau05.mat'));
F5 = load(fullfile([fl '_tau05'], 'flow.mat'));
C5 = load(fullfile(cb, 'cant_tau05.mat'));

% ---- forcing coefficients (Section 4.1)
N.eta = struct('shape_identification', S.settings.eta, 'flow', F.eta, 'cantilever', C.settings.eta);

% ---- Tables 3-5: full order and CL-ROM at matched length
N.table3_shape_identification = comparison(S.fom, S.compact);
N.table4_flow                 = comparison(F.fom, F.compact);
N.table5_cantilever           = comparison(C.fom, C.compact);

% ---- quantities quoted in the text of Sections 4.2-4.4
N.text.shape_identification = struct( ...
    'shape_error_fom', S.fom.shape_error, 'shape_error_clrom', S.compact.shape_error, ...
    'phase_max_difference', max(abs(S.fom.phi(:) - S.compact.phi(:))), ...
    'thresholded_mismatch', mean((S.fom.phi(:) > .5) ~= (S.compact.phi(:) > .5)), ...
    'clrom_own_stop_updates', S.compact_own.iters, 'final_width_cells', S.compact.width);
N.text.flow = struct( ...
    'drag_reduction', 1 - F.fom.final_J/F.fom.J(1), ...
    'phase_max_difference', max(abs(F.fom.phi(:) - F.compact.phi(:))), ...
    'thresholded_mismatch', mean((F.fom.phi(:) > .5) ~= (F.compact.phi(:) > .5)), ...
    'final_width_cells', F.compact.width);
N.text.cantilever = struct( ...
    'objective_reduction', 1 - C.fom.final_J/C.fom.J(1), ...
    'phase_max_difference', max(abs(C.fom.phi(:) - C.compact.phi(:))), ...
    'thresholded_mismatch', mean((C.fom.phi(:) > .5) ~= (C.compact.phi(:) > .5)), ...
    'final_width_cells', C.compact.width);

% ---- Table 6: same-design audit of residual-screened candidates
N.table6_audit = struct('shape_identification', audit(S.compact_own.es, S.compact_own.n_rom), ...
    'flow', audit(F.audit.es, F.audit.n_rom), ...
    'cantilever', audit(C.compact_own.es, C.compact_own.n_rom));

% ---- Table 7: fixed-epsilon mesh refinement
N.table7_refinement = struct( ...
    'shape_identification', refinement(fullfile(R, 'scalar_manufactured_2d.csv')), ...
    'flow', refinement(fullfile(R, 'flow_mesh_fixed_epsilon.csv')), ...
    'cantilever', refinement(fullfile(R, 'elastic_manufactured_2d.csv')));

% ---- Table 8: design time step
N.table8_tau = struct( ...
    'shape_identification_tau001', tau_row(S.fom, S.compact), ...
    'shape_identification_tau005', tau_row(S5.fom, S5.compact), ...
    'flow_tau001', tau_row(F.fom, F.compact), 'flow_tau005', tau_row(F5.fom, F5.compact), ...
    'cantilever_tau001', tau_row(C.fom, C.compact), 'cantilever_tau005', tau_row(C5.fom, C5.compact));

print_tables(N);
fid = fopen(outfile, 'w'); fprintf(fid, '%s', jsonencode(N, 'PrettyPrint', true)); fclose(fid);
fprintf('\nWrote %s\n', outfile);
end

% =====================================================================
function t = comparison(f, r)
tail = @(x) x(max(1, numel(x)-49):end);
fm = mean(tail(f.J)); rm = mean(tail(r.J));
t = struct('updates', f.iters, ...
    'fom', struct('solves', f.n_fom, 'factorizations', f.nfact, 'mean_J', fm, 'seconds', f.time_with_setup), ...
    'clrom', struct('updates', r.iters, 'full_order', r.n_fom, 'verifications', r.n_verify, ...
        'reduced', r.n_rom, 'corrected', sum(r.source == 2), 'factorizations', r.nfact, ...
        'mean_J', rm, 'seconds', r.time_with_setup), ...
    'speedup', f.time_with_setup/r.time_with_setup, 'relative_mean_J_gap', abs(rm - fm)/abs(fm));
end

function a = audit(es, accepted)
if isfield(es, 'err_s_m'), state = [es.err_s_m]; else, state = [es.err_s]; end
full = [es.err_s];                                  % the indicator measures the full state
obj = abs([es.J] - [es.JF])./max(abs([es.JF]), 1e-30);
a = struct('samples', numel(es), 'accepted_candidates', accepted, ...
    'max_state_error', max(state), 'max_objective_error', max(obj), ...
    'indicator_below_state_error', mean([es.eta_s]./max(full, realmin) < 1), ...
    'median_objective_effectivity', median(abs([es.etaJ])./max(abs([es.J] - [es.JF]), realmin)));
end

function r = refinement(file)
T = readtable(file); T = T(strcmp(T.study, 'fixed-epsilon'), :);
E = T.fixed_epsilon_error(1:3);
r = struct('epsilon', T.epsilon(1), 'h0', T.h(2), 'errors', E(:).', ...
    'rates', log2(E(1:2)./E(2:3)).');
end

function t = tau_row(f, r)
tail = @(x) x(max(1, numel(x)-49):end);
fm = mean(tail(f.J)); rm = mean(tail(r.J));
t = struct('updates', f.iters, 'fom_seconds', f.time_with_setup, 'clrom_seconds', r.time_with_setup, ...
    'speedup', f.time_with_setup/r.time_with_setup, 'mean_J', fm, ...
    'relative_mean_J_gap', abs(rm - fm)/abs(fm), 'band_cells', r.width);
end

function print_tables(N)
names = {'table3_shape_identification', 'table4_flow', 'table5_cantilever'};
fprintf('\nTables 3-5: method, updates, FOM/ROM solves, corr., fact., mean J, time (s), speedup, |dJ|/J\n');
for i = 1:3
    t = N.(names{i});
    fprintf('%s\n  full order %6d  %6d / --      --  %4d  %.6e  %8.1f\n', names{i}, t.updates, ...
        t.fom.solves, t.fom.factorizations, t.fom.mean_J, t.fom.seconds);
    fprintf('  CL-ROM     %6d  %d+%d / %d  %4d  %4d  %.6e  %8.1f  %.2f  %.3e\n', t.clrom.updates, ...
        t.clrom.full_order, t.clrom.verifications, t.clrom.reduced, t.clrom.corrected, ...
        t.clrom.factorizations, t.clrom.mean_J, t.clrom.seconds, t.speedup, t.relative_mean_J_gap);
end
fprintf('\nTable 6: problem, samples, max state error, max objective error | indicator < error, objective effectivity\n');
for nm = fieldnames(N.table6_audit).'
    a = N.table6_audit.(nm{1});
    fprintf('  %-22s %4d  %.1e  %.1e | %.0f%%  %.2f\n', nm{1}, a.samples, a.max_state_error, ...
        a.max_objective_error, 100*a.indicator_below_state_error, a.median_objective_effectivity);
end
fprintf('\nTable 7: model, epsilon, h0, E_2h0, E_h0, E_h0/2, rates\n');
for nm = fieldnames(N.table7_refinement).'
    r = N.table7_refinement.(nm{1});
    fprintf('  %-22s %.3g  %.3g  %.2e  %.2e  %.2e  %.2f/%.2f\n', nm{1}, r.epsilon, r.h0, r.errors, r.rates);
end
fprintf('\nTable 8: row, updates, FOM (s), CL-ROM (s), speedup, mean J, |dJ|/J, band\n');
for nm = fieldnames(N.table8_tau).'
    t = N.table8_tau.(nm{1});
    fprintf('  %-30s %6d  %7.1f  %6.1f  %.2f  %.6e  %.1e  %.2fh\n', nm{1}, t.updates, t.fom_seconds, ...
        t.clrom_seconds, t.speedup, t.mean_J, t.relative_mean_J_gap, t.band_cells);
end
fprintf('\neta: %.4f %.4f %.4f\n', N.eta.shape_identification, N.eta.flow, N.eta.cantilever);
s = N.text.shape_identification; f = N.text.flow; c = N.text.cantilever;
fprintf('shape identification: shape error %.4e / %.4e, phase diff %.2e, mismatch %.1e, own stop %d\n', ...
    s.shape_error_fom, s.shape_error_clrom, s.phase_max_difference, s.thresholded_mismatch, s.clrom_own_stop_updates);
fprintf('flow: drag reduction %.1f%%, phase diff %.2e, mismatch %.1e\n', 100*f.drag_reduction, ...
    f.phase_max_difference, f.thresholded_mismatch);
fprintf('cantilever: objective reduction %.1f%%, phase diff %.2e, mismatch %.1e\n', 100*c.objective_reduction, ...
    c.phase_max_difference, c.thresholded_mismatch);
end
