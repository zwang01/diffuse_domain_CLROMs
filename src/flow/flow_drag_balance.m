function [F, dFdU] = flow_drag_balance(par, S, U)
% Streamwise force on the body from an x-momentum balance over the FIXED
% channel [x0,xR] x [y0,y0+1], using boundary values of the MAC solution only:
%
%   F = int_in (p + u^2) dy - int_out (p + u^2) dy + nu * int_walls du/dn_out dx
%
% par.drag_form='traction' uses the complete outer-boundary momentum flux.
% The outlet condition nu*du/dn-p=0 cancels outlet pressure and normal
% viscous stress; inlet normal viscous stress is retained. Wall friction
% enters with +nu*int du/dn (negative for forward channel flow).
% The absent/'legacy' option reproduces the archived truncated flux above.
% The boundary diagnostic depends on the diffuse solution and is not, by
% itself, a guarantee of sharp-interface drag accuracy.
%
% Inlet/outlet pressures are extrapolated linearly to the boundary faces; wall
% shear uses the no-slip ghost convention u_ghost = -u_1, i.e. du/dn = -2u_1/h.
% dFdU is the exact gradient (the u^2 flux terms make it state dependent).

Nx = par.Nx; Ny = par.Ny; h = par.h; nu = par.nu;
Nu = S.Nu; Nv = S.Nv;
iu = @(i,j) i + (j-1)*(Nx+1);
ip = @(i,j) Nu + Nv + i + (j-1)*Nx;

u = reshape(U(1:Nu), Nx+1, Ny);
p = reshape(U(Nu+Nv+1:end), Nx, Ny);

p_in  = 1.5*p(1,:)  - 0.5*p(2,:);          % faces x = x0 and x = xR
p_out = 1.5*p(Nx,:) - 0.5*p(Nx-1,:);
cw = ones(Nx+1,1); cw([1 end]) = 0.5;       % trapezoid weights along the walls

complete = isfield(par,'drag_form') && strcmp(par.drag_form,'traction');
if isfield(par,'drag_form')
    assert(ismember(par.drag_form,{'legacy','traction'}),'Unknown drag form');
end
F = h*sum(p_in) + h*sum(u(1,:).^2-u(Nx+1,:).^2) ...
    - 2*nu*sum(cw.' .* (u(:,Ny).' + u(:,1).'));
if complete
    du_in = (-3*u(1,:)+4*u(2,:)-u(3,:))/(2*h);
    F = F - nu*h*sum(du_in);
else
    F = F - h*sum(p_out);
end

if nargout < 2, return, end
dFdU = zeros(S.Ntot,1);
j = 1:Ny;
dFdU(ip(1,j))    = dFdU(ip(1,j))    + 1.5*h;
dFdU(ip(2,j))    = dFdU(ip(2,j))    - 0.5*h;
if ~complete
    dFdU(ip(Nx,j))   = dFdU(ip(Nx,j))   - 1.5*h;
    dFdU(ip(Nx-1,j)) = dFdU(ip(Nx-1,j)) + 0.5*h;
end
dFdU(iu(1,j))    = dFdU(iu(1,j))    + 2*h*u(1,j).';
dFdU(iu(Nx+1,j)) = dFdU(iu(Nx+1,j)) - 2*h*u(Nx+1,j).';
if complete
    dFdU(iu(1,j)) = dFdU(iu(1,j)) + 1.5*nu;
    dFdU(iu(2,j)) = dFdU(iu(2,j)) - 2*nu;
    dFdU(iu(3,j)) = dFdU(iu(3,j)) + 0.5*nu;
end
i = (1:Nx+1).';
dFdU(iu(i,Ny)) = dFdU(iu(i,Ny)) - 2*nu*cw;
dFdU(iu(i,1))  = dFdU(iu(i,1))  - 2*nu*cw;
end
