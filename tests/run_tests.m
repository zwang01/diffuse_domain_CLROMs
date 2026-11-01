function run_tests()
% RUN_TESTS  Code verification behind the checks quoted in the manuscript.
%   test_common_balance          common design settings and update invariants
%   test_flow_traction_objective drag functional, discrete adjoint gradient,
%                                Taylor test (Section 4.3)
%   test_elastic_element         element-block reduced operators of the cantilever
%   test_cantilever_checks       patch, beam, sensitivity, and update checks
%                                (Section 4.4)
here = fileparts(mfilename('fullpath'));
addpath(here); addpath(genpath(fullfile(fileparts(here), 'src')));
tests = {@test_common_balance, @test_flow_traction_objective, ...
    @test_elastic_element, @test_cantilever_checks};
for i = 1:numel(tests)
    fprintf('\n==== %s\n', func2str(tests{i}));
    t0 = tic; tests{i}(); fprintf('(%.1f s)\n', toc(t0));
end
fprintf('\nAll tests passed.\n');
end
