function run_cant_tau05_common(max_steps)
% RUN_CANT_TAU05_COMMON  Cantilever at tau=0.05 (Appendix E, Table 8).
if nargin<1, max_steps=12000; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
rdir=fullfile(root,'results','common_balance_r03');
out=fullfile(rdir,'cant_tau05.mat');
prob=balanced_ddm_problem('cant',struct('tau',.05));
if isfile(out), S=load(out); else
    S=struct('case_name','cant_tau05','rho',.3,'tau',.05, ...
        'max_steps',max_steps,'settings',prob.info.balance);
end
cfg=struct('max_itr',max_steps-1,'stop',true,'tol_dphi',1e-5, ...
    'verbose',true,'print_every',100,'save_phi_at', ...
    unique([0,100:250:max_steps-1,max_steps-1]));
if ~isfield(S,'fom')
    c=cfg; c.tol_eta=0; c.fallback='precond';
    S.fom=finish_run(tr_optimize(prob,c),prob);
    save(out,'-struct','S','-v7.3');
end
if ~isfield(S,'compact')
    trial=balanced_ddm_problem('cant',struct('tau',.05, ...
        'case_options',struct('element_ops',true)));
    c=cfg; c.max_itr=S.fom.iters-1; c.stop=false; c.fallback='precond';
    c.compact=true; c.pod_tol=1e-10;
    S.compact=finish_run(tr_optimize(trial,c),trial);
    save(out,'-struct','S','-v7.3');
end
nf=numel(S.fom.J); nr=numel(S.compact.J);
fmean=mean(S.fom.J(max(1,nf-49):nf));
rmean=mean(S.compact.J(max(1,nr-49):nr));
fprintf(['TAU05: FOM %d updates, Jmean %.12g, audited %.12g, %.3f s; ' ...
    'CL-ROM %.3f s, speedup %.6f, mean gap %.6g, audited %.12g.\n'], ...
    S.fom.iters,fmean,S.fom.final_J,S.fom.time_with_setup, ...
    S.compact.time_with_setup,S.fom.time_with_setup/S.compact.time_with_setup, ...
    abs(rmean-fmean)/abs(fmean),S.compact.final_J);
end

function r=finish_run(r,prob)
r.time_with_setup=r.time+prob.info.balance.setup_seconds;
r.setup_seconds=prob.info.balance.setup_seconds;
audit=prob.fom(r.phi,[],true);
r.final_J=audit.J; r.final_aux=audit.aux;
r.width=iface_band(r.phi,prob.info.par.h);
B=r.phi>.1 & r.phi<.9; Fp=r.phi.*(1-r.phi).*(.5-r.phi);
q=prob.forcing(r.phi,audit.sens);
r.final_balance=prob.info.balance.eta*norm(q(B))/max(norm(Fp(B)),realmin);
end
