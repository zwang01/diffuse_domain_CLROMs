function run_scalar_flow_tau05_common(cases,max_steps)
% RUN_SCALAR_FLOW_TAU05_COMMON  Shape identification at tau=0.05 (Appendix E,
% Table 8). The flow rows of Table 8 come from run_submission_flow; the
% 'flow' option here reproduces an earlier flow model and is not used.
if nargin<1, cases={'scalar'}; end
if ischar(cases), cases={cases}; end
if nargin<2, max_steps=12000; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
rdir=fullfile(root,'results','common_balance_r03');
for k=1:numel(cases)
    name=cases{k};
    assert(ismember(name,{'scalar','flow'}),'Expected scalar or flow.');
    out=fullfile(rdir,[name '_tau05.mat']);
    prob=balanced_ddm_problem(name,struct('tau',.05));
    if isfile(out), S=load(out); else
        S=struct('case_name',[name '_tau05'],'rho',.3,'tau',.05, ...
            'max_steps',max_steps,'settings',prob.info.balance);
    end
    cfg=struct('max_itr',max_steps-1,'stop',true,'tol_dphi',1e-5, ...
        'verbose',true,'print_every',100,'save_phi_at', ...
        unique([0,100:250:max_steps-1,max_steps-1]));
    if ~isfield(S,'fom')
        c=cfg; c.tol_eta=0; c.fallback='precond';
        S.fom=finish_run(tr_optimize(prob,c),prob,name);
        save(out,'-struct','S','-v7.3');
    end
    if ~isfield(S,'compact')
        trial=balanced_ddm_problem(name,struct('tau',.05));
        c=cfg; c.max_itr=S.fom.iters-1; c.stop=false;
        c.compact=true; c.pod_tol=1e-10;
        if strcmp(name,'flow')
            c.pod_tol=1e-12; c.correction_method='gmres';
            c.corr_tol=1e-3; c.corr_maxit=8; c.max_corrections=1;
        end
        S.compact=finish_run(tr_optimize(trial,c),trial,name);
        save(out,'-struct','S','-v7.3');
    end
    report_result(S,name);
end
write_csv(rdir,cases);
end

function r=finish_run(r,prob,name)
r.time_with_setup=r.time+prob.info.balance.setup_seconds;
r.setup_seconds=prob.info.balance.setup_seconds;
audit=prob.fom(r.phi,[],true);
r.final_J=audit.J; r.final_aux=audit.aux;
if strcmp(name,'scalar'), h=prob.info.D.hx; else, h=prob.info.par.h; end
r.width=iface_band(r.phi,h);
B=r.phi>.1 & r.phi<.9; Fp=r.phi.*(1-r.phi).*(.5-r.phi);
q=prob.forcing(r.phi,audit.sens);
r.final_balance=prob.info.balance.eta*norm(q(B))/max(norm(Fp(B)),realmin);
if strcmp(name,'scalar')
    D=prob.info.D; target=inpolygon(D.X,D.Y,D.xG,D.yG);
    r.shape_error=sum(abs(double(r.phi>.5)-target),'all')*D.hx*D.hy;
end
end

function report_result(S,name)
nf=numel(S.fom.J); nr=numel(S.compact.J);
fmean=mean(S.fom.J(max(1,nf-49):nf));
rmean=mean(S.compact.J(max(1,nr-49):nr));
fprintf(['TAU05 %s: FOM %d updates, Jmean %.12g, audited %.12g, %.3f s; ' ...
    'CL-ROM %.3f s, speedup %.6f, mean gap %.6g, audited %.12g.\n'], ...
    name,S.fom.iters,fmean,S.fom.final_J,S.fom.time_with_setup, ...
    S.compact.time_with_setup,S.fom.time_with_setup/S.compact.time_with_setup, ...
    abs(rmean-fmean)/max(abs(fmean),realmin),S.compact.final_J);
end

function write_csv(rdir,cases)
rows=struct([]);
for k=1:numel(cases)
    name=cases{k}; S=load(fullfile(rdir,[name '_tau05.mat']));
    if ~isfield(S,'fom') || ~isfield(S,'compact'), continue, end
    nf=numel(S.fom.J); nr=numel(S.compact.J);
    fm=mean(S.fom.J(max(1,nf-49):nf));
    rm=mean(S.compact.J(max(1,nr-49):nr));
    row=struct('case_name',name,'tau',S.tau,'eta',S.settings.eta, ...
        'iterations',S.fom.iters,'stop_reason',S.fom.stop_reason, ...
        'J_mean50',fm,'J_audited',S.fom.final_J,'width_cells',S.fom.width, ...
        'FOM_seconds',S.fom.time_with_setup,'FOM_factorizations',S.fom.nfact, ...
        'CLROM_seconds',S.compact.time_with_setup, ...
        'CLROM_speedup',S.fom.time_with_setup/S.compact.time_with_setup, ...
        'CLROM_full_solves',S.compact.n_fom,'CLROM_verifications',S.compact.n_verify, ...
        'CLROM_corrected',sum(S.compact.source==2), ...
        'CLROM_mean50_gap',abs(rm-fm)/max(abs(fm),realmin), ...
        'CLROM_J_audited',S.compact.final_J,'CLROM_width_cells',S.compact.width);
    if strcmp(name,'scalar')
        row.FOM_shape_error=S.fom.shape_error;
        row.CLROM_shape_error=S.compact.shape_error;
    else
        row.FOM_shape_error=NaN; row.CLROM_shape_error=NaN;
    end
    rows=[rows;row]; %#ok<AGROW>
end
if ~isempty(rows)
    writetable(struct2table(rows),fullfile(rdir,'scalar_flow_tau05.csv'));
end
end
