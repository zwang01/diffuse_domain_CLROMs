function [tol_s, tol_a] = corr_tols(rom, R)
% CORR_TOLS  Relative residual reductions a 'gmres' correction aims at: the
% reduction the screen requires of this attempt (R.corr_target_s/_a, set by
% tr_optimize from the calibrated indicators and tau_eta), bounded below by
% rom.corr_tol so that a loose target never asks for less than the standard
% tolerance would allow and a tight one never for more than corr_tol.
if isfield(rom, 'corr_tol'), tol_s = rom.corr_tol; else, tol_s = 1e-3; end
tol_a = tol_s;
if isfield(R, 'corr_target_s') && ~isempty(R.corr_target_s), tol_s = max(tol_s, min(R.corr_target_s, 0.5)); end
if isfield(R, 'corr_target_a') && ~isempty(R.corr_target_a), tol_a = max(tol_a, min(R.corr_target_a, 0.5)); end
end
