function [K, F, phiu, phiv] = ddm_stokes_assemble(par, S, phi)
% Assemble the phi-weighted (diffuse-domain) Stokes operator and right-hand
% side from the phi-independent pieces built by ddm_stokes_base:
%
%   momentum rows :  phi_f .* (viscous + grad p)  +  alpha*(1-phi_f)^q u
%   continuity    :  div(phi_f .* u)  -  delta*p      (uniform nugget)
%
% phi_f is the phase field interpolated to the velocity faces. By default,
% alpha=eps_ddm^{-2} and q=1, recovering the published model exactly.
% Optional par.penalty_coeff and par.penalty_power permit fixed-geometry
% formulation studies. par.delta_p is a UNIFORM pressure nugget
% (default 1e-8/h) that removes the singularity of the weighted continuity
% equation where phi -> 0.
%
% The nugget must be phase-field INDEPENDENT. A weight such as delta*(1-phi)
% looks more targeted -- it vanishes in the fluid -- but it is active through
% the whole interfacial band, where it acts as a spurious mass source: with
% (1-phi)^4 the outflow flux of a Poiseuille test is wrong by 20%. A uniform
% nugget leaves the solution invariant over at least eight orders of
% magnitude (1e-14 to 1e-6 all reproduce max|u| = 0.3847 and the exact flux
% to 0.03%), and being design independent it also leaves the affine structure
% used by the reduced-order model untouched.

Nx = par.Nx; Ny = par.Ny;
Nu = S.Nu; Nv = S.Nv; Np = S.Np; Ntot = S.Ntot;
if isfield(par,'eps_ddm'), epsd = par.eps_ddm; else, epsd = 2*par.h; end
if isfield(par,'delta_p'), dlt = par.delta_p; else, dlt = 1e-8/par.h; end
if isfield(par,'penalty_coeff'), alpha = par.penalty_coeff; else, alpha = epsd^(-2); end
if isfield(par,'penalty_power'), penalty_power = par.penalty_power; else, penalty_power = 1; end
assert(isfinite(alpha) && alpha>0 && isfinite(penalty_power) && penalty_power>=1);

% phase field at u- and v-faces (same averaging as the Brinkman coefficient)
phiu = zeros(Nx+1,Ny);
phiu(2:Nx,:) = 0.5*(phi(1:Nx-1,:) + phi(2:Nx,:));
phiu(1,:) = phi(1,:); phiu(Nx+1,:) = phi(Nx,:);
phiv = zeros(Nx,Ny+1);
phiv(:,2:Ny) = 0.5*(phi(:,1:Ny-1) + phi(:,2:Ny));
phiv(:,1) = phi(:,1); phiv(:,Ny+1) = phi(:,Ny);

wmom = [S.momu(:); S.momv(:); false(Np,1)];        % rows carrying the PDE
phif = [phiu(:); phiv(:); ones(Np,1)];             % face weights
roww = wmom .* phif;                               % row scaling on momentum rows

K = S.Kbc ...
  + spdiags(roww, 0, Ntot, Ntot) * S.Kmom ...                     % phi*(visc + grad p)
  + spdiags(wmom .* alpha .* (1 - phif).^penalty_power, 0, Ntot, Ntot) ...
  + S.Kdiv * spdiags(phif, 0, Ntot, Ntot);                        % div(phi u)

if dlt > 0        % uniform pressure nugget (see header)
    reg = [zeros(Nu+Nv,1); -dlt*ones(Np,1)];
    K = K + spdiags(reg, 0, Ntot, Ntot);
end

F = S.Fbc + roww .* S.Fmom;      % Dirichlet data enters the scaled viscous stencil
end
