function phi = cantilever_phi0(par, eps_ac, opt)
% CANTILEVER_PHI0  Perforated-plate initial design for the 2D cantilever.
%
% Takezawa et al. (2010) start their cantilever example from the perforated
% plate of their Fig. 6(b): a staggered (hexagonal) array of circular holes in
% the 2 x 1 design domain. The layout is given only pictorially in the paper,
% so the parameters below were read off the figure; they matter, because the
% gated Allen-Cahn flow has no hole-nucleation mechanism and the paper states
% explicitly that the initial shape influences which optimum is reached.
%
% Rows alternate between nc and nc-1 holes; the first and last rows sit on the
% bottom and top edges and are therefore cut in half by the domain boundary.
% phi = 1 is material, phi = 0 is void, with a tanh profile of width eps_ac
% across each hole boundary so that the interfaces exist from the first step
% (a uniform phi = 1 is a fixed point of the flow: the gate phi^2(1-phi)^2
% vanishes identically and nothing can move).
if nargin < 3, opt = struct(); end
if ~isfield(opt,'nrow'), opt.nrow = 5;    end
if ~isfield(opt,'ncol'), opt.ncol = 5;    end
if ~isfield(opt,'rad'),  opt.rad  = 0.15; end

xc = ((1:par.Nx)-0.5)*par.h;                 % element centres, domain [0,2]x[0,1]
yc = ((1:par.Ny)-0.5)*par.h;
[X,Y] = ndgrid(xc, yc);

Lx = par.Nx*par.h; Ly = par.Ny*par.h;
dy = Ly/(opt.nrow-1);
d  = inf(size(X));                           % signed distance to nearest hole
for j = 1:opt.nrow
    yj = (j-1)*dy;
    if mod(j,2) == 1, n = opt.ncol;   x0 = Lx/(2*n);
    else,             n = opt.ncol-1; x0 = Lx/(2*(n+1)) + Lx/(2*n)*0;  x0 = Lx/(2*opt.ncol) + Lx/opt.ncol/2;
    end
    if mod(j,2) == 1, xs = x0 + (0:n-1)*(Lx/n);
    else,             xs = (1:n)*(Lx/(n+1));
    end
    for m = 1:numel(xs)
        d = min(d, hypot(X-xs(m), Y-yj) - opt.rad);
    end
end
% d < 0 inside a hole (void). phi = 1 in material, 0 in void.
phi = 0.5*(1 + tanh(3*d/eps_ac));
phi = min(max(phi,0),1);
end
