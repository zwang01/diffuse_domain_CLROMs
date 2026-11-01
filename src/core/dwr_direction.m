function [rdir, inspan] = dwr_direction(rs, PhiA)
% DWR_DIRECTION  Enrichment direction of the dual-weighted estimate (paper,
% [eq:dwr_flow]): the state residual rs with its component in the orthonormal
% adjoint basis PhiA removed, then normalized, so that [PhiA, rdir] is
% orthonormal. inspan = true when rs lies in span(PhiA) to working precision;
% the caller then uses the unenriched adjoint.
rdir = rs - PhiA*(PhiA.'*rs);
nr = norm(rdir);
inspan = nr <= 1e-8*max(norm(rs), 1e-300);
if inspan, rdir = zeros(size(rs)); else, rdir = rdir/nr; end
end
