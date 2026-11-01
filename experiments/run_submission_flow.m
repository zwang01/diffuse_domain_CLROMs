function run_submission_flow(max_steps)
% RUN_SUBMISSION_FLOW  Drag minimization of Section 4.3 (Tables 4 and 6) and
% its tau=0.05 row in Table 8. For tau = 0.01 and 0.05 it runs the
% preconditioned full-order optimization to its stopping test and CL-ROM for
% the same number of updates; for tau = 0.01 it also runs the same-design
% audit. Results go to results/flow_submission_traction[_tau05]/flow.mat.
if nargin<1,max_steps=12000;end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
for tau=[.01,.05]
    tag='flow_submission_traction';
    if tau==.05,tag=[tag '_tau05'];end
    out=fullfile(root,'results',tag);if ~isfolder(out),mkdir(out);end
    file=fullfile(out,'flow.mat');
    if isfile(file),data=load(file);else,data=struct('tau',tau,'max_steps',max_steps);end
    assert(data.tau==tau && data.max_steps==max_steps,'Saved campaign differs');
    cfg=struct('max_itr',max_steps-1,'stop',true,'tol_dphi',1e-5, ...
        'verbose',true,'print_every',500,'save_phi_at',unique([0,250:500:max_steps-1]));
    for item={'fom','compact'}
        name=item{1};if isfield(data,name),continue,end
        p=balanced_ddm_problem('flow',struct('rho',.3,'tau',tau, ...
            'case_options',struct('penalty_C',64,'penalty_power',2,'drag_form','traction')));
        c=cfg;
        if strcmp(name,'fom'),c.tol_eta=0;c.fallback='precond';
        else
            c.max_itr=data.fom.iters-1;c.stop=false;c.compact=true;c.pod_tol=1e-12;
            c.correction_method='gmres';c.corr_tol=1e-3;c.corr_maxit=8;c.max_corrections=1;
        end
        fprintf('SUBMISSION FLOW tau=%g %s\n',tau,name);
        r=tr_optimize(p,c);r.setup_seconds=p.info.balance.setup_seconds;
        r.time_with_setup=r.time+r.setup_seconds;
        a=p.fom(r.phi,[],true);r.final_J=a.J;r.width=iface_band(r.phi,p.info.par.h);
        data.(name)=r;data.eta=p.info.balance.eta;
        save(file,'-struct','data','-v7.3');
    end
    f=data.fom;r=data.compact;tail=max(1,f.iters-49):f.iters;
    summary=table(f.iters,data.eta,mean(f.J(tail)),mean(r.J(tail)), ...
        abs(mean(f.J(tail))-mean(r.J(tail)))/abs(mean(f.J(tail))), ...
        f.time_with_setup,r.time_with_setup,f.time_with_setup/r.time_with_setup, ...
        f.nfact,r.nfact,r.n_fom,r.n_verify,r.n_rom,sum(r.source==2),r.width,f.width, ...
        max(abs(f.phi(:)-r.phi(:))),mean((f.phi(:)>.5)~=(r.phi(:)>.5)), ...
        'VariableNames',{'updates','eta','mean_J_fom','mean_J_clrom','mean_J_gap', ...
        'time_fom','time_clrom','speedup','fact_fom','fact_clrom','full_solves', ...
        'verifications','rom_solves','corrected','width_clrom','width_fom', ...
        'phase_max_difference','thresholded_difference'});
    writetable(summary,fullfile(out,'summary.csv'));disp(summary);
    if tau==.01 && ~isfield(data,'audit')
        p=balanced_ddm_problem('flow',struct('rho',.3,'tau',tau, ...
            'case_options',struct('penalty_C',64,'penalty_power',2,'drag_form','traction')));
        c=cfg;c.compact=true;c.pod_tol=1e-12;c.correction_method='gmres';
        c.corr_tol=1e-3;c.corr_maxit=8;c.max_corrections=1;c.errstudy=true;c.es_every=50;
        data.audit=tr_optimize(p,c);a=p.fom(data.audit.phi,[],true);data.audit.final_J=a.J;
        save(file,'-struct','data','-v7.3');
    end
    if isfield(data,'audit')
        es=data.audit.es; state=[es.err_s_m]; obj=abs([es.J]-[es.JF])./max(abs([es.JF]),1e-30);
        fullstate=[es.err_s];
        cal=[es.eta_s]./max(fullstate,realmin);
        audit=table(data.audit.iters,numel(es),median(state),max(state),median(obj),max(obj), ...
            median([es.etaE_s]./max(fullstate,realmin)), ...
            median([es.etaP_s]./max(fullstate,realmin)),median(cal),mean(cal<1), ...
            median(abs([es.etaJ])./max(abs([es.J]-[es.JF]),realmin)),max([es.err_g]), ...
            median(fullstate),max(fullstate), ...
            'VariableNames',{'updates','samples','state_median','state_max','objective_median', ...
            'objective_max','raw_effectivity','preconditioned_effectivity','calibrated_effectivity', ...
            'below_fraction','objective_effectivity','forcing_max','full_state_median','full_state_max'});
        writetable(audit,fullfile(out,'audit_summary.csv'));disp(audit);
    end
end
end
