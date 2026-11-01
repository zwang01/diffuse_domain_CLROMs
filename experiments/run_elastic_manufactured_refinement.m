function T = run_elastic_manufactured_refinement()
% Two-dimensional traction-free disk using the paper's Q1 element assembly.
% Sharp exact displacement: u=(1-r^2/R^2)^3 (x,y) inside the disk, zero
% outside.  Its stress vanishes at r=R, so the disk boundary is traction-free.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
hs = [0.02,0.01,0.005,0.0025];
box = 0.4; R = 0.25; E = 1; nuP = 0.3; kmin = 1e-5; power = 3;
T = table();
for study = 1:2
    fields = cell(numel(hs),1);
    for q = 1:numel(hs)
        h = hs(q); epsd = 0.04;
        if study == 2, epsd = 4*h; end
        n = round(2*box/h); assert(abs(n*h-2*box)<1e-12);
        par = struct('Nx',n,'Ny',n,'h',h,'E',E,'nu',nuP,'p',power,'kmin',kmin,'load',0);
        S = elast_base(par);
        xc = -box + ((1:n)-0.5)*h;
        [Xc,Yc] = ndgrid(xc,xc);
        phi = 0.5*(1-tanh(3*(hypot(Xc,Yc)-R)/epsd));
        K = elast_assemble(S,kmin+(1-kmin)*phi(:).^power);
        F = manufactured_load(S,box,h,epsd,R,E,nuP);
        [Ii,Jj] = ndgrid(1:n+1,1:n+1);
        boundary = Ii==1 | Ii==n+1 | Jj==1 | Jj==n+1;
        boundary = find(boundary(:));
        fixed = sort([2*boundary-1;2*boundary]);
        free = setdiff((1:S.ndof)',fixed);
        tic;
        U = zeros(S.ndof,1);
        U(free) = K(free,free)\F(free);
        elapsed = toc;
        xn = -box+(0:n)*h;
        [Xn,Yn] = ndgrid(xn,xn);
        r2 = Xn.^2+Yn.^2;
        z = max(1-r2/R^2,0);
        exactx = z.^3.*Xn; exacty = z.^3.*Yn;
        ux = reshape(U(1:2:end),n+1,n+1);
        uy = reshape(U(2:2:end),n+1,n+1);
        fields{q} = struct('ux',ux,'uy',uy,'exactx',exactx,'exacty',exacty,'r2',r2);
        mask = r2 < R^2;
        sharp = norm([ux(mask)-exactx(mask);uy(mask)-exacty(mask)]) / ...
            norm([exactx(mask);exacty(mask)]);
        name = 'fixed-epsilon'; if study==2, name = 'epsilon=4h'; end
        row = table(string(name),h,epsd,epsd/h,sharp,NaN,elapsed, ...
            'VariableNames',{'study','h','epsilon','eps_over_h','sharp_error', ...
            'fixed_epsilon_error','seconds'});
        T = [T;row]; %#ok<AGROW>
        fprintf('ELASTIC %s h=%g eps=%g sharp=%.5e time=%.1fs\n', ...
            name,h,epsd,sharp,elapsed);
        clear K S U F;
    end
    if study == 1
        ref = fields{end};
        rows = height(T)-numel(hs)+(1:numel(hs));
        for q=1:numel(hs)-1
            ratio = round(hs(q)/hs(end));
            ids = 1:ratio:size(ref.ux,1);
            f = fields{q}; mask = f.r2<R^2;
            rx = ref.ux(ids,ids); ry = ref.uy(ids,ids);
            T.fixed_epsilon_error(rows(q)) = norm([f.ux(mask)-rx(mask);f.uy(mask)-ry(mask)]) / ...
                norm([rx(mask);ry(mask)]);
        end
    end
    writetable(T,fullfile(root,'results','elastic_manufactured_2d.csv'));
end
end

function F = manufactured_load(S,box,h,epsd,R,E,nuP)
% Two-by-two Gauss integration of phi*f against the Q1 basis.
n = round(2*box/h);
[I,J] = ndgrid(1:n,1:n);
cx = -box+(I(:)-0.5)*h; cy = -box+(J(:)-0.5)*h;
F = zeros(S.ndof,1);
g = 1/sqrt(3); points = [-g -g;g -g;g g;-g g];
M = E/(1-nuP^2);
for t=1:4
    xi=points(t,1); eta=points(t,2);
    x=cx+0.5*h*xi; y=cy+0.5*h*eta; s=x.^2+y.^2;
    z=max(1-s/R^2,0);
    qp=-3*z.^2/R^2; qpp=6*z/R^4;
    coeff=-M*(8*qp+4*s.*qpp);
    coeff(s>=R^2)=0;
    phase=0.5*(1-tanh(3*(sqrt(s)-R)/epsd));
    fx=phase.*coeff.*x; fy=phase.*coeff.*y;
    Nv=0.25*[(1-xi)*(1-eta),(1+xi)*(1-eta), ...
        (1+xi)*(1+eta),(1-xi)*(1+eta)];
    for a=1:4
        F=F+accumarray(S.edof(:,2*a-1),Nv(a)*fx*(h^2/4),[S.ndof,1]);
        F=F+accumarray(S.edof(:,2*a),Nv(a)*fy*(h^2/4),[S.ndof,1]);
    end
end
end
