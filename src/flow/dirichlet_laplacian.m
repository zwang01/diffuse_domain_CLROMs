function [L, bvec] = dirichlet_laplacian(Nx, Ny, h, bcL, bcR, bcB, bcT)
% Cell-centered 5-point Laplacian on an Nx x Ny grid with DIRICHLET values
% prescribed on the boundary faces, eliminated via the ghost convention
% phi_ghost = 2*phi_bc - phi_1 (boundary face midway between ghost and
% first cell). Returns the sparse operator L (acting on interior/cell
% unknowns) and the affine boundary contribution bvec, so that the
% continuous Laplacian is approximated by  L*phi + bvec.
%
% bcL, bcR : Ny-vectors of phi values on the left/right boundary faces
% bcB, bcT : Nx-vectors of phi values on the bottom/top boundary faces

n = Nx*Ny;
idx = @(i,j) i + (j-1)*Nx;
II = zeros(5*n,1); JJ = II; VV = II; nz = 0;
bvec = zeros(Nx,Ny);
    function add(r,c,v), nz=nz+1; II(nz)=r; JJ(nz)=c; VV(nz)=v; end
for j = 1:Ny
    for i = 1:Nx
        r = idx(i,j); d = 0;
        if i > 1,  add(r, idx(i-1,j), 1/h^2); d = d-1;
        else,      d = d-3; bvec(i,j) = bvec(i,j) + 2*bcL(j)/h^2; end
        if i < Nx, add(r, idx(i+1,j), 1/h^2); d = d-1;
        else,      d = d-3; bvec(i,j) = bvec(i,j) + 2*bcR(j)/h^2; end
        if j > 1,  add(r, idx(i,j-1), 1/h^2); d = d-1;
        else,      d = d-3; bvec(i,j) = bvec(i,j) + 2*bcB(i)/h^2; end
        if j < Ny, add(r, idx(i,j+1), 1/h^2); d = d-1;
        else,      d = d-3; bvec(i,j) = bvec(i,j) + 2*bcT(i)/h^2; end
        add(r, r, d/h^2);
    end
end
L = sparse(II(1:nz), JJ(1:nz), VV(1:nz), n, n);
bvec = bvec(:);
end
