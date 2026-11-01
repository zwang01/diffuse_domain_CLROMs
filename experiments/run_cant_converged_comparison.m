function run_cant_converged_comparison()
% RUN_CANT_CONVERGED_COMPARISON  Cantilever of Section 4.4 (Tables 5 and 6).
% Joins the 12,000-update full-order run of run_common_balance and its
% continuation (continue_cant_fom) into one 15,581-update record, then runs
% CL-ROM for that length ('compact') and under its own stopping test with
% the same-design audit ('compact_own'). Saves cant_converged.mat.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
rdir=fullfile(root,'results','common_balance_r03');
base=load(fullfile(rdir,'cant.mat'));
C=load(fullfile(rdir,'cant_fom_continuation.mat'));
out=fullfile(rdir,'cant_converged.mat');
if isfile(out), S=load(out); else, S=struct(); end
if ~isfield(S,'fom')
    S.case_name='cant'; S.rho=base.rho; S.settings=base.settings;
    S.fom=combine_fom(base.fom,C.continuation);
    S.max_steps=S.fom.iters;
    save(out,'-struct','S','-v7.3');
end
% Refresh terminal diagnostics because the combined record initially inherits
% width/balance fields from the 12,000-update segment.
check=balanced_ddm_problem('cant');
audit=check.fom(S.fom.phi,[],true);
S.fom.final_J=audit.J; S.fom.final_aux=audit.aux;
S.fom.width=iface_band(S.fom.phi,check.info.par.h);
B=S.fom.phi>.1 & S.fom.phi<.9;
Fp=S.fom.phi.*(1-S.fom.phi).*(.5-S.fom.phi);
q=check.forcing(S.fom.phi,audit.sens);
S.fom.final_balance=check.info.balance.eta*norm(q(B))/max(norm(Fp(B)),realmin);
save(out,'-struct','S','-v7.3');
cfg=struct('max_itr',S.fom.iters-1,'stop',false,'tol_dphi',1e-5, ...
    'fallback','precond', ...
    'verbose',true,'print_every',250,'save_phi_at', ...
    unique([0,250:500:S.fom.iters-1,S.fom.iters-1]));
for item={'compact','compact_own'}
    method=item{1}; if isfield(S,method), continue, end
    prob=balanced_ddm_problem('cant',struct('case_options',struct('element_ops',true)));
    c=cfg; c.compact=true; c.pod_tol=1e-10;
    if strcmp(method,'compact_own')
        c.stop=true; c.errstudy=true; c.es_every=50;
    end
    fprintf('RUN converged-length cantilever %s\n',method);
    r=tr_optimize(prob,c);
    r.time_with_setup=r.time+prob.info.balance.setup_seconds;
    r.setup_seconds=prob.info.balance.setup_seconds;
    audit=prob.fom(r.phi,[],true);
    r.final_J=audit.J; r.final_aux=audit.aux;
    r.width=iface_band(r.phi,prob.info.par.h);
    B=r.phi>.1 & r.phi<.9; Fp=r.phi.*(1-r.phi).*(.5-r.phi);
    q=prob.forcing(r.phi,audit.sens);
    r.final_balance=prob.info.balance.eta*norm(q(B))/max(norm(Fp(B)),realmin);
    S.(method)=r; save(out,'-struct','S','-v7.3');
    fprintf('DONE %s: %d updates, audited J %.12g, %.3f s, %s\n', ...
        method,r.iters,r.final_J,r.time_with_setup,r.stop_reason);
end
end

function r=combine_fom(a,b)
r=a;
hist={'J','aux','eta','eta_s','eta_a','eta_g','eg','etaJ','tol','mode', ...
    'rej','since','r','rA','dphi','t','source','correction_steps', ...
    'step_scale','step_reason'};
for i=1:numel(hist)
    f=hist{i};
    if isfield(a,f) && isfield(b,f), r.(f)=[a.(f);b.(f)]; end
end
r.iters=a.iters+b.iters; r.k_last=r.iters-1;
r.stop_reason=b.stop_reason; r.phi0=a.phi0; r.phi=b.phi;
r.final_J=b.final_J; r.final_aux=b.final_aux;
r.time=a.time+b.time;
r.time_with_setup=a.time_with_setup+b.time;
r.nfact=a.nfact+b.nfact;
counts={'n_enrich','n_enrich_skipped','n_stop_verify','n_fom','n_rom', ...
    'n_verify','n_verify_dwr','n_relax','n_pc','n_pc_fail','n_refresh', ...
    'n_second','n_correct_screen','n_correct_objective', ...
    'n_correct_screen_attempt','n_correct_objective_attempt'};
for i=1:numel(counts)
    f=counts{i}; if isfield(a,f) && isfield(b,f), r.(f)=a.(f)+b.(f); end
end
if isfield(a,'pc_its') && isfield(b,'pc_its')
    r.pc_its=[a.pc_its(:);b.pc_its(:)];
end
if isfield(a,'T') && isfield(b,'T')
    names=fieldnames(a.T);
    for i=1:numel(names), f=names{i}; r.T.(f)=a.T.(f)+b.T.(f); end
end
r.continuation_restart=true;
r.continuation_added_updates=b.iters;
r.continuation_note=['Restarted at the saved 12,000-update phase field; ' ...
    'the continuation includes one restart factorization.'];
end
