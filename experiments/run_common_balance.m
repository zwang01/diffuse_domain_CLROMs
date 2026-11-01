function run_common_balance(cases,max_steps,tag)
% RUN_COMMON_BALANCE  Shape identification (Section 4.2, Tables 3 and 6) and
% the first 12,000 cantilever updates (Section 4.4), under the common design
% settings of Section 4.1.
%
% For each case it runs the preconditioned full-order optimization to its
% stopping test ('fom'), CL-ROM for the same number of updates ('compact'),
% and CL-ROM under its own stopping test with the same-design audit
% ('compact_own', excluded from timings). Runs are sequential to avoid CPU
% competition; each completed leg is saved to results/<tag>/<case>.mat and
% skipped on restart. Timings include the common setup and calibration.
% The flow case of the paper is run by run_submission_flow.
if nargin<1, cases={'scalar','cant'}; end
if ischar(cases), cases={cases}; end
if nargin<2, max_steps=12000; end
if nargin<3, tag='common_balance_r03'; end
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(fullfile(root,'src')));
outdir=fullfile(root,'results',tag); if ~isfolder(outdir), mkdir(outdir); end
cfg=struct('max_itr',max_steps-1,'stop',true,'tol_dphi',1e-5,'verbose',true, ...
    'print_every',250,'save_phi_at',unique([0,250:500:max_steps-1,max_steps-1]));
for k=1:numel(cases)
    name=cases{k}; file=fullfile(outdir,[name '.mat']);
    p=balanced_ddm_problem(name);
    S=struct('case_name',name,'rho',.3,'max_steps',max_steps,'settings',p.info.balance);
    if isfile(file)
        S=load(file); assert(S.max_steps==max_steps && S.rho==.3);
        assert(abs(S.settings.eta-p.info.balance.eta)<1e-10*max(1,S.settings.eta));
    end
    fprintf('\nCOMMON %s eta %.9g initial R %.6f width %.3fh setup %.3fs\n', ...
        name,S.settings.eta,S.settings.initial_ratio,S.settings.initial_width,S.settings.setup_seconds);
    for leg={'fom','compact','compact_own'}
        method=leg{1}; if isfield(S,method), continue, end
        c=cfg; trial=p;
        if strcmp(method,'fom')
            c.tol_eta=0; c.fallback='precond';
        else
            c.fallback='precond'; % use the same full-order solver for ROM fallbacks and verifications
            if ~strcmp(method,'compact_own'), c.max_itr=S.fom.iters-1; c.stop=false; end
            c.compact=true; c.pod_tol=1e-10;
            if strcmp(name,'flow')
                c.pod_tol=1e-12; c.correction_method='gmres';
                c.corr_tol=1e-3; c.corr_maxit=8; c.max_corrections=1;
            elseif strcmp(name,'cant')
                trial=balanced_ddm_problem(name,struct('case_options',struct('element_ops',true)));
            end
            if strcmp(method,'compact_own'), c.errstudy=true; c.es_every=50; end
        end
        fprintf('RUN %s %s\n',name,method);
        r=tr_optimize(trial,c);
        r.time_with_setup=r.time+trial.info.balance.setup_seconds;
        r.setup_seconds=trial.info.balance.setup_seconds;
        audit=trial.fom(r.phi,[],true);
        r.final_J=audit.J; r.final_aux=audit.aux;
        if strcmp(name,'scalar'), h=trial.info.D.hx; else, h=trial.info.par.h; end
        r.width=iface_band(r.phi,h);
        B=r.phi>.1 & r.phi<.9; Fp=r.phi.*(1-r.phi).*(.5-r.phi);
        q=trial.forcing(r.phi,audit.sens);
        r.final_balance=trial.info.balance.eta*norm(q(B))/max(norm(Fp(B)),realmin);
        if strcmp(name,'scalar')
            D=trial.info.D; target=inpolygon(D.X,D.Y,D.xG,D.yG);
            r.shape_error=sum(abs(double(r.phi>.5)-target),'all')*D.hx*D.hy;
        end
        S.(method)=r; save(file,'-struct','S','-v7.3');
        m=max(1,r.iters-49):r.iters;
        fprintf('DONE %s %s: %d iterations, mean J %.9g, final audit %.9g, width %.3fh, %.1fs (%s)\n', ...
            name,method,r.iters,mean(r.J(m)),r.final_J,r.width,r.time_with_setup,r.stop_reason);
        write_summary(outdir);
    end
end
end

function write_summary(outdir)
rows=struct([]);
for name={'scalar','flow','cant'}
    file=fullfile(outdir,[name{1} '.mat']); if ~isfile(file), continue,end
    S=load(file); if ~isfield(S,'fom'),continue,end
    for leg={'fom','compact','compact_own'}
        method=leg{1}; if ~isfield(S,method),continue,end
        r=S.(method); tail=max(1,r.iters-49):r.iters;
        same=r.iters==S.fom.iters;
        row=struct('case_name',name{1},'method',method,'rho',S.rho,'eta',S.settings.eta, ...
            'iterations',r.iters,'J_initial',r.J(1),'J_mean50',mean(r.J(tail)), ...
            'J_audited',r.final_J,'width_cells',r.width,'final_balance',r.final_balance, ...
            'seconds',r.time_with_setup,'speedup_matched',NaN,'mean50_gap',NaN, ...
            'max_history_gap',NaN,'shape_error',NaN,'full_solves',r.n_fom,'verifications',r.n_verify, ...
            'reduced',r.n_rom,'corrected',sum(r.source==2),'factorizations',r.nfact, ...
            'tail_dphi_max',max(r.dphi(tail)),'stop_reason',r.stop_reason);
        if same
            if ~strcmp(method,'compact_own'), row.speedup_matched=S.fom.time_with_setup/r.time_with_setup; end
            fmean=mean(S.fom.J(tail)); row.mean50_gap=abs(row.J_mean50-fmean)/abs(fmean);
            row.max_history_gap=max(abs(r.J-S.fom.J)./max(abs(S.fom.J),1e-30));
        end
        if isfield(r,'shape_error'), row.shape_error=r.shape_error; end
        rows=[rows;row]; %#ok<AGROW>
    end
end
writetable(struct2table(rows),fullfile(outdir,'summary.csv'));
end
