function T = run_flow_manufactured_refinement(model,outname)
% Two-dimensional manufactured swirling flow around a fixed circular body.
% The sharp velocity is (-g(r^2)y,g(r^2)x), p=0.  Its support is inside
% r<B and it vanishes at the obstacle r=R.  The analytic Navier--Stokes
% forcing is added to the paper's phi-weighted MAC operator.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
if nargin<1 || isempty(model),model=struct();end
if nargin<2 || isempty(outname),outname='flow_manufactured_2d.csv';end
if ~isfield(model,'penalty_power'),model.penalty_power=1;end
hs = [0.05,0.025,0.0125,0.00625];
box = 0.75; R = 0.25; B = 0.6; nu = 0.02/40;
T = table();
studies=1:2;
if isfield(model,'fixed_only') && model.fixed_only, studies=1; end
for study = studies
    fields = cell(numel(hs),1);
    for q = 1:numel(hs)
        h = hs(q); epsd = 0.1;
        if study==2, epsd=4*h; end
        n=round(2*box/h); assert(abs(n*h-2*box)<1e-12);
        par=struct('Nx',n,'Ny',n,'h',h,'x0',-box,'y0',-box, ...
            'nu',nu,'eps_ddm',epsd,'delta_p',1e-8/h);
        if isfield(model,'fixed_delta_p'),par.delta_p=model.fixed_delta_p;end
        if isfield(model,'penalty_C'),par.penalty_coeff=model.penalty_C*nu/epsd^2;
        else,par.penalty_coeff=epsd^(-2);end
        par.penalty_power=model.penalty_power;
        par.ubc=@(x,y) zeros(size(x+y));
        par.vbc=@(x,y) zeros(size(x+y));
        par.openRight=false(n,1); par.openBottom=false(n,1);
        xc=-box+((1:n)-0.5)*h;
        [Xc,Yc]=ndgrid(xc,xc);
        phi=0.5*(1+tanh(3*(hypot(Xc,Yc)-R)/epsd));
        S=ddm_stokes_base(par);
        [K,F,phiu,phiv]=ddm_stokes_assemble(par,S,phi);
        xu=-box+(0:n)*h; yu=xc;
        xv=xc; yv=xu;
        [Xu,Yu]=ndgrid(xu,yu); [Xv,Yv]=ndgrid(xv,yv);
        [eu,~,fu,~]=swirl(Xu,Yu,R,B,nu);
        [~,ev,~,fv]=swirl(Xv,Yv,R,B,nu);
        F(1:S.Nu)=F(1:S.Nu)+reshape(S.momu.*phiu.*fu,S.Nu,1);
        F(S.Nu+1:S.Nu+S.Nv)=F(S.Nu+1:S.Nu+S.Nv)+ ...
            reshape(S.momv.*phiv.*fv,S.Nv,1);
        tic;
        fac=fact_lu(K,[0.1 0]);
        [U,ok]=fact_solve(fac,K,F,false);
        assert(ok,'Manufactured Stokes solve failed at h=%g',h);
        scale=max(norm(F),1e-14);
        for it=1:20
            [N,dN]=ddm_nse_convection(par,S,U,phiu,phiv);
            residual=K*U+N-F; nr=norm(residual)/scale;
            if nr<1e-10, break, end
            A=K+dN; fac=fact_lu(A,[0.1 0]);
            [step,ok]=fact_solve(fac,A,residual,false);
            assert(ok,'Manufactured Newton solve failed at h=%g',h);
            alpha=1;
            for bt=1:8
                trial=U-alpha*step;
                Nt=ddm_nse_convection(par,S,trial,phiu,phiv);
                if norm(K*trial+Nt-F)/scale < nr, break, end
                alpha=alpha/2;
            end
            U=U-alpha*step;
        end
        N=ddm_nse_convection(par,S,U,phiu,phiv);
        nr=norm(K*U+N-F)/scale;
        assert(nr<1e-8,'Manufactured Newton residual %.3e at h=%g',nr,h);
        elapsed=toc;
        u=reshape(U(1:S.Nu),n+1,n);
        v=reshape(U(S.Nu+1:S.Nu+S.Nv),n,n+1);
        masku=hypot(Xu,Yu)>R; maskv=hypot(Xv,Yv)>R;
        sharp=norm([u(masku)-eu(masku);v(maskv)-ev(maskv)]) / ...
            norm([eu(masku);ev(maskv)]);
        fields{q}=struct('u',u,'v',v,'xu',xu,'yu',yu,'xv',xv,'yv',yv, ...
            'masku',masku,'maskv',maskv);
        name='fixed-epsilon'; if study==2, name='epsilon=4h'; end
        row=table(string(name),h,epsd,epsd/h,sharp,NaN,nr,it,elapsed,par.delta_p, ...
            'VariableNames',{'study','h','epsilon','eps_over_h', ...
            'sharp_error','fixed_epsilon_error','residual','newton_steps','seconds','delta_p'});
        T=[T;row]; %#ok<AGROW>
        fprintf('FLOW %s h=%g eps=%g sharp=%.5e residual=%.2e time=%.1fs\n', ...
            name,h,epsd,sharp,nr,elapsed);
        clear K F S U A fac N dN;
    end
    if study==1
        ref=fields{end};
        uref=griddedInterpolant({ref.xu,ref.yu},ref.u,'linear','none');
        vref=griddedInterpolant({ref.xv,ref.yv},ref.v,'linear','none');
        rows=height(T)-numel(hs)+(1:numel(hs));
        for q=1:numel(hs)-1
            f=fields{q};
            [X,Y]=ndgrid(f.xu,f.yu); ru=uref(X,Y);
            [X,Y]=ndgrid(f.xv,f.yv); rv=vref(X,Y);
            T.fixed_epsilon_error(rows(q))=norm([f.u(f.masku)-ru(f.masku); ...
                f.v(f.maskv)-rv(f.maskv)]) / norm([ru(f.masku);rv(f.maskv)]);
        end
    end
    writetable(T,fullfile(root,'results',outname));
end
end

function [u,v,fx,fy]=swirl(x,y,R,B,nu)
s=x.^2+y.^2; active=s>R^2 & s<B^2;
a=s-R^2; b=B^2-s;
g=zeros(size(s)); gp=g; gpp=g;
g(active)=a(active).*b(active).^3;
gp(active)=b(active).^3-3*a(active).*b(active).^2;
gpp(active)=6*b(active).*(a(active)-b(active));
lap=8*gp+4*s.*gpp;
u=-g.*y; v=g.*x;
fx=nu*lap.*y-g.^2.*x;
fy=-nu*lap.*x-g.^2.*y;
end
