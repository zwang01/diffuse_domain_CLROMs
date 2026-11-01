function test_common_balance()
root=fileparts(fileparts(mfilename('fullpath'))); addpath(genpath(fullfile(root,'src')));
for name={'scalar','flow','cant'}
    c=name{1}; o=struct();
    switch c
        case 'scalar', o.h=.008;
        case 'flow', o.Nx=40;
        case 'cant', o.Nx=60; o.Ny=30;
    end
    p=balanced_ddm_problem(c,struct('case_options',o));
    assert(abs(p.info.balance.initial_ratio-.3)<1e-13);
    assert(p.info.opts.Kin==5 && p.info.opts.tau==.01 && p.info.opts.eps_fac==4);
    assert(all(p.phi0(:)>=0 & p.phi0(:)<=1));
    a=p.fom(p.phi0,[],true); ph=p.update(p.phi0,a.sens);
    assert(all(isfinite(ph(:))) && min(ph(:))>=-1e-12 && max(ph(:))<=1+1e-12);
    if strcmp(c,'scalar')
        opt=p.info.opts; opt.ac_band=false; exact=scalar_surface_problem(opt);
        phfull=exact.update(p.phi0,a.sens);
        difference=max(abs(ph(:)-phfull(:)));
        assert(difference<1e-10,'Scalar band update disagrees with full-grid gated update.');
        fprintf('Scalar band/full-grid update difference %.3e\n',difference);
    elseif strcmp(c,'flow')
        vol=sum(1-ph(:))*p.info.par.h^2;
        assert(abs(vol-p.info.opts.Vsolid)<1e-12);
    end
    exact=p.fom(ph,a.U,true);
    [pc,info]=p.pc(ph,a.U,a.W,a.Kref,struct('tol',1e-10,'maxit',100,'refresh_its',100));
    assert(info.ok,'Preconditioned solve failed after the first balanced step.');
    gap=abs(pc.J-exact.J)/max(abs(exact.J),1e-30);
    assert(gap<1e-7,'Retained-factor and fresh solves disagree.');
    fprintf('%s eta %.9g initial width %.3fh R %.15g PC objective gap %.3e\n', ...
        c,p.info.balance.eta,p.info.balance.initial_width,p.info.balance.initial_ratio,gap);
end
fprintf('COMMON BALANCE CHECKS PASSED\n');
end
