function [update, stats] = cached_ac_factory(L, shape, o)
% Cached narrow-band Allen--Cahn updates for scalar and volume-constrained
% adapters. Outside values are frozen, never rounded to zero/one. A refreshed
% band or a diffuse buffer edge rebuilds the sparse block. Linear solves use
% warm starts and a contraction-based infinity-norm error test.
o=set_defaults(o,struct('tau',.01,'kappa',1,'Kin',5,'beta',1, ...
    'active_tol',1e-12,'dil',2,'refresh',20,'solve_tol',1e-10, ...
    'freeze',false(shape),'volume',[],'gate_norm',false,'clip',false,'sweep_policy','check'));
assert(ismember(o.sweep_policy,{'check','bound'}));
assert(o.solve_tol>0 && isfinite(o.solve_tol));
C=[]; calls=0; builds=0; sweeps=0; max_active=0;
update=@step; stats=@get_stats;
    function s=get_stats()
        s=struct('calls',calls,'builds',builds,'sweeps',sweeps,'max_active',max_active);
    end
    function p=step(p,s,scale)
        if nargin<3, scale=1; end
        assert(scale>0 && isfinite(scale));
        pv=p(:); sv=s(:); tk=o.tau*scale*o.kappa; dt=o.tau*scale;
        refresh=isempty(C) || mod(calls,max(o.refresh,1))==0;
        if ~refresh
            edge=pv(C.edge); refresh=any(edge.*(1-edge)>o.active_tol);
        end
        if refresh
            m=p.*(1-p)>o.active_tol;
            m=m | (conv2(double(p>.5),ones(3),'same')>0 & conv2(double(p<=.5),ones(3),'same')>0);
            m=conv2(double(m),ones(2*o.dil+1),'same')>0; m=m & ~o.freeze;
            B=find(m);
        else, B=C.B; end
        if isempty(B), calls=calls+1; return, end
        if isempty(C) || ~isequal(B,C.B) || tk~=C.tk
            Lbb=L(B,B); dg=full(diag(Lbb));
            [~,cols]=find(L(B,:)); ext=setdiff(unique(cols),B);
            % Nodes adjacent to a non-frozen exterior node form the buffer.
            exterior=~reshape(ismember((1:numel(pv))',B),shape) & ~o.freeze;
            near=conv2(double(exterior),[0 1 0;1 0 1;0 1 0],'same')>0;
            C=struct('B',B,'tk',tk,'diag',1-tk*dg, ...
                'off',tk*(Lbb-spdiags(dg,0,numel(B),numel(B))), ...
                'ext',ext,'coupling',tk*L(B,ext),'edge',B(near(B)));
            C.rho=max(sum(abs(C.off),2)./C.diag);
            assert(C.rho<1,'Jacobi update requires a contracting band operator');
            builds=builds+1;
        end
        v=pv(B); outside_sum=sum(pv)-sum(v); cpl=C.coupling*pv(C.ext);
        global_norm=norm(sv)+1e-10; pb=zeros(size(v));
        for it=1:o.Kin
            Fw=v.*(1-v).*(.5-v); gate=v.^2.*(1-v).^2;
            gs=gate.*sv(B);
            if o.gate_norm, den=norm(gs)+1e-14; else, den=global_norm; end
            rhs=v+dt*(-Fw-o.beta*gs/den)+cpl;
            pa=solve(rhs,v);
            if isempty(o.volume), v=pa;
            else
                pb=solve(dt*gate,pb); target=o.volume-outside_sum;
                sb=sum(pb);
                if abs(sb)>1e-12
                    lambda=(target-sum(pa))/sb;
                    v=pa+lambda*pb;
                    if any(v<0 | v>1)
                        lambda=volume_root(pa,pb,target,lambda);
                        v=min(max(pa+lambda*pb,0),1);
                    end
                else, v=pa; end
            end
            if o.clip, v=min(max(v,0),1); end
        end
        p(B)=v; calls=calls+1; max_active=max(max_active,numel(B));
    end
    function x=solve(rhs,x)
        if strcmp(o.sweep_policy,'bound')
            % If delta_1=x_1-x_0 and the Jacobi contraction is rho, then
            % ||x_* - x_j||_inf <= rho^j ||delta_1||_inf/(1-rho).
            % One initial difference therefore determines all sweeps.
            xn=(rhs+C.off*x)./C.diag; first=max(abs(xn-x)); x=xn;
            assert(isfinite(first),'Nonfinite cached AC iterate');
            j=1;
            if first>o.solve_tol*(1-C.rho) && C.rho>0
                j=max(1,ceil(log(o.solve_tol*(1-C.rho)/first)/log(C.rho)));
            end
            assert(j<=60,'Cached AC solve needs more than 60 sweeps');
            for q=2:j, x=(rhs+C.off*x)./C.diag; end
            sweeps=sweeps+j; return
        end
        for j=1:60
            xn=(rhs+C.off*x)./C.diag;
            done=max(abs(xn-x))<=o.solve_tol*(1-C.rho);
            x=xn; if done, break, end
        end
        assert(done,'Cached AC solve did not reach its requested tolerance');
        sweeps=sweeps+j;
    end
end

function lambda=volume_root(a,b,V,lambda)
f=@(z)sum(min(max(a+z*b,0),1))-V;
lo=lambda; hi=lambda; step=max(1,abs(lambda));
for k=1:60, if f(lo)<=0, break, end; lo=lo-step; step=2*step; end
step=max(1,abs(lambda));
for k=1:60, if f(hi)>=0, break, end; hi=hi+step; step=2*step; end
assert(f(lo)<=0 && f(hi)>=0,'Cannot bracket the volume multiplier');
for k=1:100
    mid=(lo+hi)/2; if f(mid)<0, lo=mid; else, hi=mid; end
    if hi-lo<=1e-14*max(1,abs(mid)), break, end
end
lambda=(lo+hi)/2;
end
