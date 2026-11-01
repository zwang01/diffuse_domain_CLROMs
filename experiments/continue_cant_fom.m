function continue_cant_fom(max_additional_updates)
% Continue the common-protocol cantilever FOM from its 12,000-update design.
% The continuation is saved separately; the original campaign is untouched.
if nargin<1, max_additional_updates=12000; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
rdir=fullfile(root,'results','common_balance_r03');
base=load(fullfile(rdir,'cant.mat'));
prob=balanced_ddm_problem('cant');
assert(abs(prob.info.balance.eta-base.settings.eta) ...
    <1e-12*max(1,base.settings.eta),'The calibrated coefficient changed.');
prob.phi0=base.fom.phi;
cfg=struct('max_itr',max_additional_updates-1,'stop',true, ...
    'tol_dphi',1e-5,'verbose',true,'print_every',250, ...
    'tol_eta',0,'fallback','precond', ...
    'save_phi_at',unique([0,250:500:max_additional_updates-1,max_additional_updates-1]));
continuation=tr_optimize(prob,cfg);
audit=prob.fom(continuation.phi,[],true);
continuation.final_J=audit.J;
continuation.final_aux=audit.aux;
continuation.total_updates=base.fom.iters+continuation.iters;
continuation.combined_time_with_setup=base.fom.time_with_setup+continuation.time;
continuation.combined_factorizations=base.fom.nfact+continuation.nfact;
continuation.combined_full_solves=base.fom.n_fom+continuation.n_fom;
continuation.base_file='cant.mat';
continuation.base_updates=base.fom.iters;
continuation.eta=base.settings.eta;
save(fullfile(rdir,'cant_fom_continuation.mat'),'continuation','-v7.3');
fprintf(['CANT FOM CONTINUATION: %d + %d = %d updates; %s; ' ...
    'terminal audited J %.12g; added time %.3f s.\n'], ...
    base.fom.iters,continuation.iters,continuation.total_updates, ...
    continuation.stop_reason,continuation.final_J,continuation.time);
end
