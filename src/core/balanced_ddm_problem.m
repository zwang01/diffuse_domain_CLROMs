function prob = balanced_ddm_problem(name, options)
% Common design protocol: epsilon=4h, tau=.01, K=5, gated Euclidean norm,
% and one initial full-order calibration with the same dimensionless rho.
% The calibrated eta is fixed for the entire trajectory. case_options only
% supplies physical/discretization choices (and operator implementations).
if nargin<2, options=struct(); end
options=set_defaults(options,struct('rho',.3,'tau',.01,'case_options',struct()));
assert(options.rho>0 && isfinite(options.rho));
assert(options.tau>0 && isfinite(options.tau));
tsetup=tic;
o=options.case_options;
o.eps_fac=4; o.tau=options.tau; o.Kin=5; o.beta=1; o.cache_ops=true;
switch name
    case 'scalar'
        if ~isfield(o,'h'), o.h=.002; end
        o.gate_norm=true; o.ac_band=true;
        make=@scalar_surface_problem;
    case 'flow'
        if ~isfield(o,'Nx'), o.Nx=80; end
        o.relax_itr=0; o.clip_pct=100; o.vol_project=true;
        make=@flow_problem;
    case 'cant'
        if ~isfield(o,'Nx'), o.Nx=200; end
        if ~isfield(o,'Ny'), o.Ny=100; end
        o.fallback='precond'; % match the FOM solver at ROM fallbacks and verifications
        make=@cantilever_problem;
    otherwise, error('Unknown balanced DDM case: %s',name);
end
seed=make(o);
phi=initial_profile(name,seed);
tcal=tic; a=seed.fom(phi,[],true); calibration_seconds=toc(tcal);
q=seed.forcing(phi,a.sens);
B=phi>.1 & phi<.9;
Fp=phi.*(1-phi).*(.5-phi);
fn=norm(Fp(B)); qn=norm(q(B));
assert(nnz(B)>0 && fn>0 && qn>0,'Initial interface must support the balance calibration.');
o.beta=options.rho*fn/qn;
prob=make(o); prob.phi0=phi;
if strcmp(name,'scalar'), h=prob.info.D.hx;
else, h=prob.info.par.h; end
balance=struct('rho',options.rho,'eta',o.beta,'F_norm_band',fn,'q_norm_band',qn, ...
    'initial_ratio',o.beta*qn/fn,'band_cells',nnz(B),'initial_width',iface_band(phi,h), ...
    'initial_J',a.J,'normalization_delta',1e-14,'calibration_seconds',calibration_seconds, ...
    'setup_seconds',toc(tsetup),'initialization','signed distance, epsilon=4h', ...
    'tau',options.tau,'K',5);
prob.info.balance=balance;
end

function phi=initial_profile(name,p)
switch name
    case 'scalar'
        D=p.info.D; X=D.X; Y=D.Y;
        vx=[-.05,.5,.75,1.1,.75,.5,-.05];
        vy=[-.06,-.06,-.05,0,.05,.06,.06];
        d=inf(size(X));
        for j=1:numel(vx)
            k=mod(j,numel(vx))+1; dx=vx(k)-vx(j); dy=vy(k)-vy(j);
            t=max(0,min(1,((X-vx(j))*dx+(Y-vy(j))*dy)/(dx^2+dy^2)));
            d=min(d,hypot(X-vx(j)-t*dx,Y-vy(j)-t*dy));
        end
        inside=inpolygon(X,Y,vx,vy); d(inside)=-d(inside);
        phi=.5*(1-tanh(3*d/D.eps));
        phi([1,end],:)=0; phi(:,[1,end])=0;
    case 'flow'
        I=p.info; par=I.par; o=I.opts; [X,Y]=ndgrid(I.xc,I.yc);
        frozen=X<par.x0+.1 | X>o.xR-.1 | Y<par.y0+.1 | Y>par.y0+.9;
        d=hypot(X,Y); lo=0; hi=.49;
        % The diffuse disk, including its tails, must have the prescribed
        % solid volume before the first optimization step.
        for j=1:70
            radius=(lo+hi)/2;
            phi=.5*(1+tanh(3*(d-radius)/(4*par.h))); phi(frozen)=1;
            if sum(1-phi(:))*par.h^2>o.Vsolid, hi=radius; else, lo=radius; end
        end
    case 'cant'
        phi=p.phi0; % distance to the union of the circular holes
end
end
