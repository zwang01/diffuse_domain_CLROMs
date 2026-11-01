function run_paper_experiments(steps)
% RUN_PAPER_EXPERIMENTS  Run every MATLAB experiment of the manuscript in
% order, then export the published numbers and the figure data.
%
%   run_paper_experiments          the published runs (several hours)
%   run_paper_experiments(steps)   cap every optimization at STEPS updates;
%                                  a workflow check that does not reproduce
%                                  the paper (use a fresh copy of the package)
%
% Each driver saves its legs under results/ and skips legs that are already
% saved, so an interrupted run can be restarted. Timed legs run one after
% another; avoid other heavy work on the machine meanwhile. The Python steps
% are listed in README.md.
if nargin < 1, steps = 12000; end
here = fileparts(mfilename('fullpath'));
addpath(here); addpath(fullfile(fileparts(here), 'figures'));

run_common_balance({'scalar', 'cant'}, steps, 'common_balance_r03');  % Sec. 4.2, first cantilever leg
continue_cant_fom(steps);                % cantilever full order to its stopping test
run_cant_converged_comparison();         % Sec. 4.4, Tables 5 and 6
run_submission_flow(steps);              % Sec. 4.3, Tables 4 and 6, flow rows of Table 8
run_scalar_flow_tau05_common('scalar', steps);   % Table 8
run_cant_tau05_common(steps);            % Table 8
run_flow_manufactured_refinement(struct('penalty_C', 64, 'penalty_power', 2, ...
    'fixed_only', true, 'fixed_delta_p', 4e-7), 'flow_mesh_fixed_epsilon.csv');  % Table 7
run_elastic_manufactured_refinement();   % Table 7
export_paper_fields();                   % data of Figures 3-8
fprintf(['\nMATLAB experiments finished. Run run_scalar_manufactured_refinement.py ' ...
    '(Table 7), then export_paper_numbers and the figure scripts.\n']);
end
