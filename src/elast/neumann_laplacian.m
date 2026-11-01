function L = neumann_laplacian(Nx, Ny, h)
% Cell-centered 5-point Laplacian on an Nx x Ny grid with homogeneous
% Neumann boundary conditions (used by the Allen-Cahn design update).
ex = ones(Nx,1); ey = ones(Ny,1);
Tx = spdiags([ex -2*ex ex], [-1 0 1], Nx, Nx);
Tx(1,1) = -1; Tx(Nx,Nx) = -1;
Ty = spdiags([ey -2*ey ey], [-1 0 1], Ny, Ny);
Ty(1,1) = -1; Ty(Ny,Ny) = -1;
L = (kron(speye(Ny), Tx) + kron(Ty, speye(Nx)))/h^2;
end
