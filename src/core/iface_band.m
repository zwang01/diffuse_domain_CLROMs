function [w, info] = iface_band(phi, h, lo, hi)
% IFACE_BAND  Width of the diffuse interface, in cells.
%
%   w = iface_band(phi, h) returns the 10-90 per cent width of the phase-field
%   transition, measured as the area of {lo < phi < hi} divided by the length
%   of the interface and by h. The interface length is obtained from the
%   co-area formula, integral |grad phi|, which needs no contour extraction and
%   handles several components.
%
%   Why not simply count cells. Counting cells with 0.01 < phi < 0.99, the
%   criterion used previously, does not detect a collapsed interface: a step
%   profile has exponential tails that fall inside that window, so it reports a
%   width of about two cells whether or not there is a genuine transition. The
%   10-90 band measures the transition itself. On the same runs the two
%   disagree sharply -- 2.00 h against 0.36 h for a profile that goes
%   0.98 -> 0.11 in one cell, and 4.45 h against 2.19 h for a true tanh.
%
%   A healthy diffuse interface of width eps = 4h with the equilibrium profile
%   phi = (1 + tanh(3d/eps))/2 has a 10-90 band of 0.73 eps = 2.93 h. Values
%   below about one cell mean the design forcing has sharpened the profile into
%   a step, at which point the interface is pinned to the lattice and can only
%   advance in whole-cell jumps.
%
%   info.area  area of the band,  info.len  interface length,
%   info.wide  the old 0.01-0.99 width, for comparison.
if nargin < 3, lo = 0.1; end
if nargin < 4, hi = 0.9; end
area = sum(phi(:) > lo & phi(:) < hi) * h^2;
[gx, gy] = gradient(phi.', h, h);
len = sum(hypot(gx(:), gy(:))) * h^2;          % co-area: integral |grad phi|
w = area / max(len, 1e-30) / h;
if nargout > 1
    wide = sum(phi(:) > 0.01 & phi(:) < 0.99) * h^2 / max(len,1e-30) / h;
    info = struct('area', area, 'len', len, 'wide', wide);
end
end
