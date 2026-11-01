function S = elast_base(par)
% ELAST_BASE  Fixed-grid Q1 plane-stress elasticity for the diffuse-domain
% (ersatz-material) design problem.
%
% Builds everything that does not depend on the design: the element stiffness
% matrix of a single square element, the element-to-dof map, the load vector
% and the constrained dofs. The design enters only through the element weights
% k(phi_e), so K(phi) = sum_e k(phi_e) K_e is affine in design-dependent
% scalars -- the property the reduced model exploits.
%
% par fields: Nx, Ny (elements), h, E, nu, and a load/support specification.
%
% Node (i,j), i = 1..Nx+1, j = 1..Ny+1, numbered n = i + (j-1)*(Nx+1);
% dofs 2n-1 (x) and 2n (y).

Nx = par.Nx; Ny = par.Ny; h = par.h;
E  = par.E;  nu = par.nu;

% ---- element stiffness, unit square scaled by h (plane stress) ------------
% 2x2 Gauss quadrature of B' D B over the element.
D = E/(1-nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
g = 1/sqrt(3); gp = [-g -g; g -g; g g; -g g];
Ke = zeros(8,8);
for q = 1:4
    xi = gp(q,1); et = gp(q,2);
    % Q1 shape-function derivatives on the reference square [-1,1]^2
    dNdxi = 0.25*[-(1-et),  (1-et), (1+et), -(1+et)];
    dNdet = 0.25*[-(1-xi), -(1+xi), (1+xi),  (1-xi)];
    % isoparametric map for a square of side h: J = (h/2) I, detJ = h^2/4
    dNdx = dNdxi*(2/h); dNdy = dNdet*(2/h); detJ = h^2/4;
    B = zeros(3,8);
    B(1,1:2:7) = dNdx;
    B(2,2:2:8) = dNdy;
    B(3,1:2:7) = dNdy; B(3,2:2:8) = dNdx;
    Ke = Ke + B.'*D*B*detJ;
end
S.Ke = Ke;

% ---- element dof map ------------------------------------------------------
% element (i,j) has corner nodes (i,j),(i+1,j),(i+1,j+1),(i,j+1)
[ei, ej] = ndgrid(1:Nx, 1:Ny);
ei = ei(:); ej = ej(:);
n1 = ei     + (ej-1)*(Nx+1);
n2 = ei + 1 + (ej-1)*(Nx+1);
n3 = ei + 1 +  ej   *(Nx+1);
n4 = ei     +  ej   *(Nx+1);
nod = [n1 n2 n3 n4];
edof = zeros(Nx*Ny, 8);
edof(:,1:2:7) = 2*nod - 1;
edof(:,2:2:8) = 2*nod;
S.edof = edof;
S.nel  = Nx*Ny;
S.ndof = 2*(Nx+1)*(Ny+1);

% ---- sparsity pattern, assembled once -------------------------------------
iK = reshape(kron(edof, ones(8,1)).', 64*S.nel, 1);
jK = reshape(kron(edof, ones(1,8)).', 64*S.nel, 1);
S.iK = iK; S.jK = jK;
S.Kevec = Ke(:);                       % 64x1, reused for every element

% ---- supports and load ----------------------------------------------------
% Left edge (i = 1) clamped; unit downward point load at the centre of the
% right edge -- the cantilever of Takezawa et al. (2010), Section 5.1.
fixedNodes = (1:(Ny+1)).'*0 + 1 + (0:Ny).'*(Nx+1);     % i = 1, all j
S.fixed = sort([2*fixedNodes-1; 2*fixedNodes]);
S.free  = setdiff((1:S.ndof).', S.fixed);

F = zeros(S.ndof,1);
jmid = round(Ny/2) + 1;                                % centre of right edge
nload = (Nx+1) + (jmid-1)*(Nx+1);
F(2*nload) = par.load;                                 % vertical (negative = down)
S.F = F;
S.nload = nload;
end
