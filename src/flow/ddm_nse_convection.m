function [N, dNdU, dNdphif] = ddm_nse_convection(par, S, U, phiu, phiv)
% Weighted convection (u.grad)(phi*u) of the diffuse-domain Navier-Stokes
% formulation, on the MAC grid, with its exact Jacobians.
%
% Writing q = phi_f .* U for the weighted velocity, the term is
%   N_u = u * d_x q_u + vbar * d_y q_u,     N_v = ubar * d_x q_v + v * d_y q_v,
% i.e. N = M(U) q with M(U) the transport operator built from the advecting
% velocity. Hence
%   dN/dphi_f = M(U) diag(U),                      (design derivative)
%   dN/dU     = M(U) diag(phi_f) + dM/dU applied to q.
% Ghost values at no-slip walls use the same convention as the viscous
% stencil, with phi extended by a zero normal derivative.
%
% With phi == 1 this reduces exactly to the unweighted convection of
% brinkman_nse_convection.m.

Nx = par.Nx; Ny = par.Ny; h = par.h;
x0 = par.x0; y0 = par.y0; xR = x0+Nx*h; yT = y0+Ny*h;
Nu_ = S.Nu; Nv_ = S.Nv;
openR = par.openRight(:); openB = par.openBottom(:);
iu = @(i,j) i + (j-1)*(Nx+1);
iv = @(i,j) Nu_ + i + (j-1)*Nx;

u = reshape(U(1:Nu_), Nx+1, Ny);
v = reshape(U(Nu_+1:Nu_+Nv_), Nx, Ny+1);
qu = phiu .* u;                     % weighted velocities
qv = phiv .* v;

N = zeros(S.Ntot,1);
needJ = nargout > 1;      % Jacobian wanted?
needP = nargout > 2;      % design derivative wanted?
Ic={}; Jc={}; Vc={}; Ip={}; Jp={}; Vp={};
    function put(R,C,V)
        if ~needJ, return, end
        Ic{end+1}=R(:); Jc{end+1}=C(:); Vc{end+1}=V(:);
    end
    function putp(R,C,V)
        if ~needP, return, end
        Ip{end+1}=R(:); Jp{end+1}=C(:); Vp{end+1}=V(:);
    end

% ---------------- u-momentum faces i = 2..Nx ----------------
ii = (2:Nx)'; xu = x0 + (ii-1)*h;
obu = false(Nx-1,1);
for m = 1:Nx-1, obu(m) = openB(ii(m)-1) || openB(min(ii(m),Nx)); end
uc = u(2:Nx,:);
vbar = 0.25*( v(1:Nx-1,1:Ny) + v(1:Nx-1,2:Ny+1) + v(2:Nx,1:Ny) + v(2:Nx,2:Ny+1) );

dqdx = (qu(3:Nx+1,:) - qu(1:Nx-1,:))/(2*h);
q_dn = zeros(Nx-1,Ny); q_dn(:,2:Ny) = qu(2:Nx,1:Ny-1);
gb = phiu(2:Nx,1).*(2*par.ubc(xu, y0*ones(Nx-1,1)) - u(2:Nx,1));
qb1 = qu(2:Nx,1); gb(obu) = qb1(obu);
q_dn(:,1) = gb;
q_up = zeros(Nx-1,Ny); q_up(:,1:Ny-1) = qu(2:Nx,2:Ny);
q_up(:,Ny) = phiu(2:Nx,Ny).*(2*par.ubc(xu, yT*ones(Nx-1,1)) - u(2:Nx,Ny));
dqdy = (q_up - q_dn)/(2*h);

[IIg,JJg] = ndgrid(2:Nx,1:Ny);
R = iu(IIg,JJg);
N(R(:)) = uc(:).*dqdx(:) + vbar(:).*dqdy(:);

% --- dN/dU : transport part M(U)diag(phi) ---
put(R, iu(IIg+1,JJg),  uc.*phiu(3:Nx+1,:)/(2*h));
put(R, iu(IIg-1,JJg), -uc.*phiu(1:Nx-1,:)/(2*h));
put(R(:,1:Ny-1), iu(IIg(:,1:Ny-1),JJg(:,1:Ny-1)+1),  vbar(:,1:Ny-1).*phiu(2:Nx,2:Ny)/(2*h));
put(R(:,2:Ny),   iu(IIg(:,2:Ny),  JJg(:,2:Ny)-1),   -vbar(:,2:Ny).*phiu(2:Nx,1:Ny-1)/(2*h));
% ghost rows: wall ghosts depend on the interior value itself
gdiag = zeros(Nx-1,Ny);
gdiag(:,1)  = (~obu).*( vbar(:,1).*phiu(2:Nx,1)/(2*h)) + obu.*(-vbar(:,1).*phiu(2:Nx,1)/(2*h));
gdiag(:,Ny) = gdiag(:,Ny) - vbar(:,Ny).*phiu(2:Nx,Ny)/(2*h);
% --- dN/dU : advecting-velocity part dM/dU ---
put(R, R, dqdx + gdiag);
put(R, iv(IIg-1,JJg),   0.25*dqdy);
put(R, iv(IIg-1,JJg+1), 0.25*dqdy);
put(R, iv(IIg,  JJg),   0.25*dqdy);
put(R, iv(IIg,  JJg+1), 0.25*dqdy);

% --- dN/dphi_f ---
putp(R, iu(IIg+1,JJg),  uc.*u(3:Nx+1,:)/(2*h));
putp(R, iu(IIg-1,JJg), -uc.*u(1:Nx-1,:)/(2*h));
putp(R(:,1:Ny-1), iu(IIg(:,1:Ny-1),JJg(:,1:Ny-1)+1),  vbar(:,1:Ny-1).*u(2:Nx,2:Ny)/(2*h));
putp(R(:,2:Ny),   iu(IIg(:,2:Ny),  JJg(:,2:Ny)-1),   -vbar(:,2:Ny).*u(2:Nx,1:Ny-1)/(2*h));
gphi = zeros(Nx-1,Ny);
gb_u = 2*par.ubc(xu, y0*ones(Nx-1,1)) - u(2:Nx,1);
ub1  = u(2:Nx,1); gb_u(obu) = ub1(obu);
gphi(:,1)  = -vbar(:,1).*gb_u/(2*h);
gphi(:,Ny) = gphi(:,Ny) + vbar(:,Ny).*(2*par.ubc(xu, yT*ones(Nx-1,1)) - u(2:Nx,Ny))/(2*h);
putp(R, R, gphi);

% ---------------- v-momentum faces j = 2..Ny ----------------
jj = (2:Ny)'; yv = y0 + (jj-1)*h;
orv = false(Ny-1,1);
for m = 1:Ny-1, orv(m) = openR(jj(m)-1) || openR(min(jj(m),Ny)); end
vc = v(:,2:Ny);
ubar = 0.25*( u(1:Nx,1:Ny-1) + u(1:Nx,2:Ny) + u(2:Nx+1,1:Ny-1) + u(2:Nx+1,2:Ny) );

dqdy2 = (qv(:,3:Ny+1) - qv(:,1:Ny-1))/(2*h);
q_lt = zeros(Nx,Ny-1); q_lt(2:Nx,:) = qv(1:Nx-1,2:Ny);
q_lt(1,:) = phiv(1,2:Ny).*(2*par.vbc(x0*ones(Ny-1,1), yv)' - v(1,2:Ny));
q_rt = zeros(Nx,Ny-1); q_rt(1:Nx-1,:) = qv(2:Nx,2:Ny);
grt = phiv(Nx,2:Ny).*(2*par.vbc(xR*ones(Ny-1,1), yv)' - v(Nx,2:Ny));
qvN = qv(Nx,2:Ny); grt(orv') = qvN(orv');
q_rt(Nx,:) = grt;
dqdx2 = (q_rt - q_lt)/(2*h);

[IIg2,JJg2] = ndgrid(1:Nx,2:Ny);
R2 = iv(IIg2,JJg2);
N(R2(:)) = ubar(:).*dqdx2(:) + vc(:).*dqdy2(:);

put(R2, iv(IIg2,JJg2+1),  vc.*phiv(:,3:Ny+1)/(2*h));
put(R2, iv(IIg2,JJg2-1), -vc.*phiv(:,1:Ny-1)/(2*h));
put(R2(2:Nx,:),   iv(IIg2(2:Nx,:)-1,  JJg2(2:Nx,:)),   -ubar(2:Nx,:).*phiv(1:Nx-1,2:Ny)/(2*h));
put(R2(1:Nx-1,:), iv(IIg2(1:Nx-1,:)+1,JJg2(1:Nx-1,:)),  ubar(1:Nx-1,:).*phiv(2:Nx,2:Ny)/(2*h));
gdiag2 = zeros(Nx,Ny-1);
gdiag2(1,:)  = ubar(1,:).*phiv(1,2:Ny)/(2*h);
gdiag2(Nx,:) = gdiag2(Nx,:) + (~orv').*(-ubar(Nx,:).*phiv(Nx,2:Ny)/(2*h)) ...
                            + ( orv').*( ubar(Nx,:).*phiv(Nx,2:Ny)/(2*h));
put(R2, R2, dqdy2 + gdiag2);
put(R2, iu(IIg2,  JJg2-1), 0.25*dqdx2);
put(R2, iu(IIg2,  JJg2),   0.25*dqdx2);
put(R2, iu(IIg2+1,JJg2-1), 0.25*dqdx2);
put(R2, iu(IIg2+1,JJg2),   0.25*dqdx2);

putp(R2, iv(IIg2,JJg2+1),  vc.*v(:,3:Ny+1)/(2*h));
putp(R2, iv(IIg2,JJg2-1), -vc.*v(:,1:Ny-1)/(2*h));
putp(R2(2:Nx,:),   iv(IIg2(2:Nx,:)-1,  JJg2(2:Nx,:)),   -ubar(2:Nx,:).*v(1:Nx-1,2:Ny)/(2*h));
putp(R2(1:Nx-1,:), iv(IIg2(1:Nx-1,:)+1,JJg2(1:Nx-1,:)),  ubar(1:Nx-1,:).*v(2:Nx,2:Ny)/(2*h));
gphi2 = zeros(Nx,Ny-1);
gphi2(1,:) = -ubar(1,:).*(2*par.vbc(x0*ones(Ny-1,1), yv)' - v(1,2:Ny))/(2*h);
grt_u = 2*par.vbc(xR*ones(Ny-1,1), yv)' - v(Nx,2:Ny);
vN = v(Nx,2:Ny); grt_u(orv') = vN(orv');
gphi2(Nx,:) = gphi2(Nx,:) + ubar(Nx,:).*grt_u/(2*h);
putp(R2, R2, gphi2);

if needJ
    dNdU = sparse(vertcat(Ic{:}), vertcat(Jc{:}), vertcat(Vc{:}), S.Ntot, S.Ntot);
else
    dNdU = [];
end
if needP
    dNdphif = sparse(vertcat(Ip{:}), vertcat(Jp{:}), vertcat(Vp{:}), S.Ntot, S.Ntot);
else
    dNdphif = [];
end
end
