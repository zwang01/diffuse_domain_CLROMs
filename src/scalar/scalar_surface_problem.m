function prob = scalar_surface_problem(opts)
% Variational Dirichlet DDM adapter for the same shape-identification data.
% State: -div(w(phi) grad T) + beta_eps^{-1} delta(phi) T = phi f,
% beta_eps=beta_scale*eps^(3/4), w=kmin+(1-kmin)phi.
% The observed state is phi*T (Dirichlet datum zero), so the box objective
% has a defined exterior extension. delta is a regularized |grad phi|; its
% complete discrete Jacobian is included in the sensitivity.
%
% The manuscript runs build this adapter through balanced_ddm_problem, which
% sets eps_fac = 4, tau = 0.01, Kin = 5 and the calibrated forcing coefficient
% (beta, the eta of the paper); the defaults listed here are not the published
% settings.
def = struct('h',4e-3,'eps_fac',4,'tau',0.01,'beta',1e3,'Kin',10, ...
    'ac_band',true,'cache_ops',false,'gate_norm',false,'beta_scale',0.1,'kmin',1e-6, ...
    'delta_smooth',1e-5,'ac_cached',false,'ac_refresh',20, ...
    'ac_solve_tol',1e-10,'local_projection',false,'ac_sweep_policy','check');
if nargin < 1, opts = struct(); end
opts = set_defaults(opts,def);
oldopts = rmfield(opts,{'beta_scale','kmin','delta_smooth','ac_cached','ac_refresh','ac_solve_tol','local_projection','ac_sweep_policy'});
old = scalar_problem(oldopts);
D = old.info.D;
D.area = D.hx*D.hy;
D.kmin = opts.kmin;
D.lambda = 1/(opts.beta_scale*D.eps^(3/4));
D.sigma = opts.delta_smooth/D.eps;
D.local_projection=opts.local_projection;
[D.G,D.Af,D.Dx,D.Dy] = operators(D.mx,D.my,D.hx,D.hy);

prob = struct();
prob.name = sprintf('scalar surface h=%g',opts.h);
prob.phi0 = old.phi0;
prob.self_adjoint = false;
prob.fallback = 'precond'; prob.refresh_its = 8;
prob.aux_names = {};
prob.fom = @(phi,Uwarm,adj_reuse) fom(D,phi);
prob.pc = @(phi,U0,W0,Kref,o) pc(D,phi,U0,W0,Kref,o);
prob.rom_new = @(o) rom_new(D,o);
prob.enrich = @(rom,U,W,varargin) enrich(D,rom,U,W);
prob.rom_step = @(phi,rom,Uwarm) rom_step(D,phi,rom);
prob.correct = @(phi,rom,R,Kref,varargin) correct(D,phi,rom,R,Kref,varargin{:});
prob.update = old.update; prob.forcing = old.forcing;
if opts.ac_cached
    a=struct('tau',D.tau,'kappa',D.kappa,'Kin',D.Kin,'beta',D.beta, ...
        'active_tol',D.ac_tol,'dil',D.ac_dil,'refresh',opts.ac_refresh,'solve_tol',opts.ac_solve_tol, ...
        'sweep_policy',opts.ac_sweep_policy,'gate_norm',opts.gate_norm);
    [up,st]=cached_ac_factory(D.L,[D.mx,D.my],a);
    prob.update=@(phi,s)cached_update(up,D,phi,s,1);
    prob.update_scaled=@(phi,s,scale)cached_update(up,D,phi,s,scale);
    prob.stats=st;
end
prob.res_s = @(U,phi) residual_s(D,U,phi);
prob.res_a = @(W,U,phi) residual_a(D,W,U,phi);
prob.jac = @(U,phi) assemble(D,phi);
prob.info = struct('opts',opts,'D',rmfield(D,{'G','Af','Dx','Dy'}));
end

function phi=cached_update(up,D,phi,s,scale)
phi(D.Iint,D.Jint)=up(phi(D.Iint,D.Jint),s(D.Iint,D.Jint),scale);
end

function [G,Af,Dx,Dy] = operators(mx,my,hx,hy)
n = mx*my; idx = @(i,j) i+(j-1)*mx;
ng = 2*(mx-1)*my+2*mx*(my-1)+2*my+2*mx;
rows = zeros(ng,1); cols = rows; gv = rows; av = rows;
q = 0; row = 0;
for j = 1:my
    for i = 1:mx-1
        row=row+1; q=q+1; rows(q)=row; cols(q)=idx(i,j); gv(q)=-sqrt(hy/hx); av(q)=0.5;
        q=q+1; rows(q)=row; cols(q)=idx(i+1,j); gv(q)=sqrt(hy/hx); av(q)=0.5;
    end
end
for j = 1:my-1
    for i = 1:mx
        row=row+1; q=q+1; rows(q)=row; cols(q)=idx(i,j); gv(q)=-sqrt(hx/hy); av(q)=0.5;
        q=q+1; rows(q)=row; cols(q)=idx(i,j+1); gv(q)=sqrt(hx/hy); av(q)=0.5;
    end
end
for j = 1:my
    row=row+1; q=q+1; rows(q)=row; cols(q)=idx(1,j); gv(q)=sqrt(hy/hx); av(q)=0.5;
    row=row+1; q=q+1; rows(q)=row; cols(q)=idx(mx,j); gv(q)=sqrt(hy/hx); av(q)=0.5;
end
for i = 1:mx
    row=row+1; q=q+1; rows(q)=row; cols(q)=idx(i,1); gv(q)=sqrt(hx/hy); av(q)=0.5;
    row=row+1; q=q+1; rows(q)=row; cols(q)=idx(i,my); gv(q)=sqrt(hx/hy); av(q)=0.5;
end
rows=rows(1:q); cols=cols(1:q); gv=gv(1:q); av=av(1:q);
G=sparse(rows,cols,gv,row,n); Af=sparse(rows,cols,av,row,n);
Dx1=spdiags([-ones(mx,1),ones(mx,1)],[-1,1],mx,mx)/(2*hx);
Dy1=spdiags([-ones(my,1),ones(my,1)],[-1,1],my,my)/(2*hy);
Dx=kron(speye(my),Dx1); Dy=kron(Dy1,speye(mx));
end

function cf = coeffs(D,phi)
p=phi(D.Iint,D.Jint); p=p(:);
cf.p=p;
cf.wf=D.kmin+(1-D.kmin)*(D.Af*p);
cf.gx=D.Dx*p; cf.gy=D.Dy*p;
cf.gnorm=sqrt(cf.gx.^2+cf.gy.^2+D.sigma^2);
cf.delta=cf.gnorm-D.sigma;
end

function [A,b,cf] = assemble(D,phi)
cf=coeffs(D,phi);
A=D.G.'*spdiags(cf.wf,0,size(D.G,1),size(D.G,1))*D.G ...
    +spdiags(D.area*D.lambda*cf.delta,0,D.n,D.n);
b=D.area*cf.p.*D.f;
end

function Y = apply_A(D,cf,V,GV)
if nargin<4 || isempty(GV), GV=D.G*V; end
Y=D.G.'*(cf.wf.*GV)+D.area*D.lambda*cf.delta.*V;
end

function [J,g,e] = objective(D,cf,T)
e=cf.p.*T-D.Tbar;
J=D.area*(e.'*e);
g=2*D.area*cf.p.*e;
end

function s = sensitivity(D,cf,T,W,e)
qdelta=-D.area*D.lambda*T.*W;
s=2*D.area*T.*e+D.area*D.f.*W ...
    -(1-D.kmin)*D.Af.'*((D.G*T).*(D.G*W)) ...
    +D.Dx.'*(qdelta.*cf.gx./cf.gnorm) ...
    +D.Dy.'*(qdelta.*cf.gy./cf.gnorm);
sfull=zeros(D.mx+2,D.my+2);
sfull(D.Iint,D.Jint)=reshape(s,D.mx,D.my);
s=sfull;
end

function out = fom(D,phi)
[A,b,cf]=assemble(D,phi); F=fact_chol(A);
T=F.solve(b); [J,g,e]=objective(D,cf,T);
W=F.solve(g);
out=struct('U',T,'W',W,'J',J,'sens',sensitivity(D,cf,T,W,e), ...
    'aux',zeros(1,0),'Kref',F);
end

function [out,info] = pc(D,phi,U0,W0,Kref,o)
[A,b,cf]=assemble(D,phi);
[T,fl,~,it]=pc_cg(A,b,U0,Kref.solve,o.tol,o.maxit);
info=struct('ok',fl==0,'its',[it 0],'refreshed',false); out=[];
if ~info.ok, return, end
[J,g,e]=objective(D,cf,T);
[W,flw,~,itw]=pc_cg(A,g,W0,Kref.solve,o.tol,o.maxit);
info.its(2)=itw; info.ok=flw==0;
if ~info.ok, return, end
Fref=Kref;
if it>o.refresh_its, Fref=fact_chol(A); info.refreshed=true; end
out=struct('U',T,'W',W,'J',J,'sens',sensitivity(D,cf,T,W,e), ...
    'aux',zeros(1,0),'Kref',Fref);
end

function r = residual_s(D,T,phi)
[A,b]=assemble(D,phi); r=b-A*T;
end

function r = residual_a(D,W,T,phi)
[A,~,cf]=assemble(D,phi); [~,g]=objective(D,cf,T); r=g-A*W;
end

function rom = rom_new(D,o)
rom=struct('SnapU',zeros(D.n,0),'Phi',zeros(D.n,0), ...
    'SnapW',zeros(D.n,0),'PhiA',zeros(D.n,0), ...
    'GPhi',zeros(size(D.G,1),0),'GPhiA',zeros(size(D.G,1),0), ...
    'r',0,'rA',0,'w_max',o.w_max,'r_max',o.r_max, ...
    'pod_tol',o.pod_tol,'lsq',o.lsq);
end

function rom = enrich(D,rom,T,W)
if ~isempty(T) || rom.r==0
    [rom.SnapU,rom.Phi,rom.r]=pod_window(rom.SnapU,T,rom.w_max,rom.r_max,rom.pod_tol);
    rom.GPhi=D.G*rom.Phi;
end
[rom.SnapW,rom.PhiA,rom.rA]=pod_window(rom.SnapW,W,rom.w_max,rom.r_max,rom.pod_tol);
rom.GPhiA=D.G*rom.PhiA;
rom.proj=[];
end

function R = rom_step(D,phi,rom)
cf=coeffs(D,phi); b=D.area*cf.p.*D.f;
Phi=rom.Phi; PhiA=rom.PhiA;
if D.local_projection
    if ~isfield(rom,'proj'), rom.proj=[]; end
    [rom.proj,Ar,ArA]=spd_project_pair(rom.proj,Phi,PhiA,rom.GPhi,rom.GPhiA,cf.wf,D.area*D.lambda*cf.delta);
    a=Ar\(Phi.'*b); T=Phi*a; rs=b-apply_A(D,cf,T);
else
    AP=apply_A(D,cf,Phi,rom.GPhi); Ar=Phi.'*AP; Ar=(Ar+Ar.')/2;
    a=Ar\(Phi.'*b); T=Phi*a; rs=b-AP*a;
end
[J,g,e]=objective(D,cf,T);
if D.local_projection
    bb=ArA\(PhiA.'*g); W=PhiA*bb; ra=g-apply_A(D,cf,W);
    ops=struct('cf',cf,'b',b,'ArA',ArA);
else
    APA=apply_A(D,cf,PhiA,rom.GPhiA); ArA=PhiA.'*APA; ArA=(ArA+ArA.')/2;
    bb=ArA\(PhiA.'*g); W=PhiA*bb; ra=g-APA*bb;
    ops=struct('cf',cf,'b',b,'APA',APA);
end
[rdir,inspan]=dwr_direction(rs,PhiA);
if inspan
    etaJ=W.'*rs;
elseif D.local_projection
    etaJ=spd_dwr_extend(PhiA,ArA,rdir,@(v)apply_A(D,cf,v),g,rs);
else
    V=[PhiA,rdir]; AV=[APA,apply_A(D,cf,rdir)]; M=V.'*AV; M=(M+M.')/2;
    etaJ=(V*(M\(V.'*g))).'*rs;
end
R=struct('U',T,'W',W,'J',J,'sens',sensitivity(D,cf,T,W,e), ...
    'aux',zeros(1,0),'etaE_s',norm(rs)/max(norm(b),1e-30), ...
    'etaE_a',norm(ra)/max(norm(g),1e-30),'rs',rs,'ra',ra, ...
    'dJdU',g,'etaJ',etaJ,'ops',ops);
if D.local_projection, R.rom=rom; end
end

function R = correct(D,phi,rom,R,Kref,mode)
if nargin<6 || isempty(mode), mode='sa'; end
if isfield(R,'ops'), ops=R.ops; cf=ops.cf; b=ops.b;
else
    cf=coeffs(D,phi); b=D.area*cf.p.*D.f;
    ops=struct('cf',cf,'b',b,'APA',apply_A(D,cf,rom.PhiA,rom.GPhiA));
end
Afun=@(v)apply_A(D,cf,v);
[tol_s,tol_a]=corr_tols(rom,R);
gm=isfield(rom,'correction_method') && strcmp(rom.correction_method,'gmres');
if strcmp(mode,'a')
    T=R.U; J=R.J; g=R.dJdU; e=cf.p.*T-D.Tbar; rs=R.rs;
    if gm, W=R.W+pc_cg(Afun,R.ra,[],Kref.solve,tol_a,rom.corr_maxit);
    else, W=R.W+Kref.solve(R.ra); end
else
    if gm, T=R.U+pc_cg(Afun,R.rs,[],Kref.solve,tol_s,rom.corr_maxit);
    else, T=R.U+Kref.solve(R.rs); end
    [J,g,e]=objective(D,cf,T);
    ra0=g-Afun(R.W);
    if gm, W=R.W+pc_cg(Afun,ra0,[],Kref.solve,tol_a,rom.corr_maxit);
    else, W=R.W+Kref.solve(ra0); end
    rs=b-Afun(T);
end
ra=g-Afun(W);
V=compact_dwr_basis(rom.PhiA,W,rs); E=V(:,rom.rA+1:end);
if isfield(ops,'ArA')
    etaJ=spd_dwr_extend(rom.PhiA,ops.ArA,E,Afun,g,rs);
else
    AV=[ops.APA,apply_A(D,cf,E)]; M=V.'*AV; M=(M+M.')/2;
    etaJ=(V*(M\(V.'*g))).'*rs;
end
R=struct('U',T,'W',W,'J',J,'sens',sensitivity(D,cf,T,W,e), ...
    'aux',zeros(1,0),'etaE_s',norm(rs)/max(norm(b),1e-30), ...
    'etaE_a',norm(ra)/max(norm(g),1e-30),'rs',rs,'ra',ra, ...
    'dJdU',g,'etaJ',etaJ,'ops',ops);
end
