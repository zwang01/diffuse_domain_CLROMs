function setup_paths()
% SETUP_PATHS  Put the package folders on the MATLAB path and create results/.
here = fileparts(mfilename('fullpath'));
dirs = {'src/core', 'src/scalar', 'src/flow', 'src/elast', 'experiments', 'figures', 'tests'};
for i = 1:numel(dirs), addpath(fullfile(here, dirs{i})); end
if ~isfolder(fullfile(here, 'results')), mkdir(fullfile(here, 'results')); end
fprintf('setup_paths: code_ddm_shapeopt on the path\n');
end
