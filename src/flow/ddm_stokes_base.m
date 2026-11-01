function S = ddm_stokes_base(par)
% Phi-independent pieces of the DIFFUSE-DOMAIN (phi-weighted) Stokes system
% on a MAC staggered grid, in the form used by Vasudevan, Wang & Ju:
%
%   -nu*(phi*Lap u) + phi*grad p + (1/eps^2)*(1-phi)*u = 0          in D,
%    div(phi*u) - delta*p = 0                                       in D,
%
% NOTE. This is the ug = 0 case of the formulation in that work, whose general
% form penalizes toward an extension ug of the immersed boundary data and
% carries a source in the mass equation,
%
%   ... + (1/eps^2)*(1-phi)*(u - ug) = 0,     div(phi*u) = ug . grad(phi).
%
% The two agree exactly when the immersed boundary is no-slip: u -> 0 in the
% solid makes div(phi*u) = phi*div(u) + u.grad(phi) -> 0, so the source
% vanishes with the penalty target. That covers the isolated-body problem used
% here, where the inflow is imposed on the FIXED boundary through ordinary
% Dirichlet rows (par.ubc, via Fmom) and never through the penalty. It does NOT
% cover a case whose openings lie on the immersed boundary -- the cannula, for
% instance -- where ug is nonzero at the interface and both omitted terms
% matter. Do not reuse this assembly for such a problem without restoring them.
%
% The pressure term is a UNIFORM nugget, not the delta*(1-phi)*p of that work:
% see ddm_stokes_assemble, where a phase-field-weighted nugget is shown to get
% the Poiseuille outflow flux wrong by 20%.
%
% i.e. the principal part is WEIGHTED by the phase field (as in the
% diffusion model of the paper), the Dirichlet condition on the immersed
% boundary is imposed by the eps^{-2} penalty, and the continuity equation
% is written for the weighted velocity phi*u.
%
% The last term of the continuity equation is a regularization that is
% active ONLY where phi < 1: in the solid the weighted divergence degenerates
% (all coefficients vanish) and the pressure would otherwise be undetermined,
% leaving the saddle system singular. Since it carries the factor (1-phi) it
% vanishes identically in the fluid and does not perturb the flow there.
%
% Returned pieces (all phi-independent), to be combined by ddm_stokes_assemble:
%   S.Kmom  momentum rows: viscous + pressure-gradient entries, UNSCALED
%   S.Fmom  momentum-row right-hand side from Dirichlet data, UNSCALED
%   S.Kbc   boundary/do-nothing rows (never weighted)
%   S.Fbc   their right-hand side
%   S.Kdiv  continuity rows: divergence stencil with unit coefficients,
%           to be applied to (phi_face .* u)
%   S.momu, S.momv  masks of interior momentum rows
%   S.E, S.e0  dissipation operators (as in brinkman_base)

Nx = par.Nx; Ny = par.Ny; h = par.h; nu = par.nu;
x0 = par.x0; y0 = par.y0;
openR = par.openRight(:); openB = par.openBottom(:);
Nu = (Nx+1)*Ny; Nv = Nx*(Ny+1); Np = Nx*Ny; Ntot = Nu+Nv+Np;
iu = @(i,j) i + (j-1)*(Nx+1);
iv = @(i,j) Nu + i + (j-1)*Nx;
ip = @(i,j) Nu + Nv + i + (j-1)*Nx;
xu = @(i) x0 + (i-1)*h;   yu = @(j) y0 + (j-0.5)*h;
xv = @(i) x0 + (i-0.5)*h; yv = @(j) y0 + (j-1)*h;
xR = x0 + Nx*h; yT = y0 + Ny*h;
obu = @(i) (i>1 && openB(i-1)) || (i<=Nx && openB(min(i,Nx)));
orv = @(j) (j>1 && openR(j-1)) || (j<=Ny && openR(min(j,Ny)));

Im=zeros(12*Ntot,1); Jm=Im; Vm=Im; nm=0;      % momentum rows
Ib=zeros(6*Ntot,1);  Jb=Ib;  Vb=Ib;  nb=0;    % boundary rows
Id=zeros(4*Np,1);    Jd=Id;  Vd=Id;  nd=0;    % continuity rows
Fmom = zeros(Ntot,1); Fbc = zeros(Ntot,1);
momu = false(Nx+1,Ny); momv = false(Nx,Ny+1);
    function am(r,c,v), nm=nm+1; Im(nm)=r; Jm(nm)=c; Vm(nm)=v; end
    function ab(r,c,v), nb=nb+1; Ib(nb)=r; Jb(nb)=c; Vb(nb)=v; end
    function ad(r,c,v), nd=nd+1; Id(nd)=r; Jd(nd)=c; Vd(nd)=v; end

% ---------------- u rows ----------------
for j = 1:Ny
    for i = 1:Nx+1
        r = iu(i,j);
        if i == 1
            ab(r,r,1); Fbc(r) = par.ubc(x0, yu(j));
        elseif i == Nx+1 && openR(j)
            ab(r, r, nu/h); ab(r, iu(Nx,j), -nu/h); ab(r, ip(Nx,j), -1);
        elseif i == Nx+1
            ab(r,r,1); Fbc(r) = par.ubc(xR, yu(j));
        else
            momu(i,j) = true; d = 4*nu/h^2;
            am(r, iu(i-1,j), -nu/h^2); am(r, iu(i+1,j), -nu/h^2);
            if j > 1, am(r, iu(i,j-1), -nu/h^2);
            elseif obu(i), d = d - nu/h^2;
            else, d = d + nu/h^2; Fmom(r) = Fmom(r) + 2*nu/h^2*par.ubc(xu(i), y0);
            end
            if j < Ny, am(r, iu(i,j+1), -nu/h^2);
            else, d = d + nu/h^2; Fmom(r) = Fmom(r) + 2*nu/h^2*par.ubc(xu(i), yT);
            end
            am(r, r, d);
            am(r, ip(i,j), 1/h); am(r, ip(i-1,j), -1/h);
        end
    end
end

% ---------------- v rows ----------------
for j = 1:Ny+1
    for i = 1:Nx
        r = iv(i,j);
        if j == 1 && openB(i)
            ab(r, iv(i,2), nu/h); ab(r, iv(i,1), -nu/h); ab(r, ip(i,1), -1);
        elseif j == 1
            ab(r,r,1); Fbc(r) = par.vbc(xv(i), y0);
        elseif j == Ny+1
            ab(r,r,1); Fbc(r) = par.vbc(xv(i), yT);
        else
            momv(i,j) = true; d = 4*nu/h^2;
            am(r, iv(i,j-1), -nu/h^2); am(r, iv(i,j+1), -nu/h^2);
            if i > 1, am(r, iv(i-1,j), -nu/h^2);
            else, d = d + nu/h^2; Fmom(r) = Fmom(r) + 2*nu/h^2*par.vbc(x0, yv(j));
            end
            if i < Nx, am(r, iv(i+1,j), -nu/h^2);
            elseif orv(j), d = d - nu/h^2;
            else, d = d + nu/h^2; Fmom(r) = Fmom(r) + 2*nu/h^2*par.vbc(xR, yv(j));
            end
            am(r, r, d);
            am(r, ip(i,j), 1/h); am(r, ip(i,j-1), -1/h);
        end
    end
end

% ---------------- continuity rows: unit-coefficient divergence ----------
for j = 1:Ny
    for i = 1:Nx
        r = ip(i,j);
        ad(r, iu(i+1,j),  1/h); ad(r, iu(i,j), -1/h);
        ad(r, iv(i,j+1),  1/h); ad(r, iv(i,j), -1/h);
    end
end

S.Kmom = sparse(Im(1:nm), Jm(1:nm), Vm(1:nm), Ntot, Ntot);
S.Kbc  = sparse(Ib(1:nb), Jb(1:nb), Vb(1:nb), Ntot, Ntot);
S.Kdiv = sparse(Id(1:nd), Jd(1:nd), Vd(1:nd), Ntot, Ntot);
S.Fmom = Fmom; S.Fbc = Fbc;
S.momu = momu; S.momv = momv;
S.Nu = Nu; S.Nv = Nv; S.Np = Np; S.Ntot = Ntot;

% ---------------- dissipation operators (identical to brinkman_base) -------
EI=zeros(8*(Nu+Nv),1); EJ=EI; EV=EI; ez=0; e0v=[]; erow=0;
    function eadd(cols, vals, shift)
        erow = erow+1;
        for kk=1:numel(cols), ez=ez+1; EI(ez)=erow; EJ(ez)=cols(kk); EV(ez)=vals(kk); end
        e0v(erow,1) = shift; %#ok<AGROW>
    end
for j=1:Ny, for i=1:Nx, eadd([iu(i+1,j), iu(i,j)], [1/h, -1/h], 0); end, end
for i=1:Nx+1
    for j=1:Ny-1, eadd([iu(i,j+1), iu(i,j)], [1/h, -1/h], 0); end
    if ~obu(i), eadd(iu(i,1), 2/h, -2*par.ubc(xu(i),y0)/h); end
    eadd(iu(i,Ny), 2/h, -2*par.ubc(xu(i),yT)/h);
end
for j=1:Ny, for i=1:Nx, eadd([iv(i,j+1), iv(i,j)], [1/h, -1/h], 0); end, end
for j=1:Ny+1
    for i=1:Nx-1, eadd([iv(i+1,j), iv(i,j)], [1/h, -1/h], 0); end
    eadd(iv(1,j), 2/h, -2*par.vbc(x0,yv(j))/h);
    if ~orv(j), eadd(iv(Nx,j), 2/h, -2*par.vbc(xR,yv(j))/h); end
end
S.E  = sparse(EI(1:ez), EJ(1:ez), EV(1:ez), erow, Nu+Nv);
S.e0 = e0v;
end
