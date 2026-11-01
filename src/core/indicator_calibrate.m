function cal = indicator_calibrate(cal, etaE_s, etaE_a, rs, ra, dJdU, Kref, Unorm, eta_s, which)
% INDICATOR_CALIBRATE  Update the calibration ratios c_s, c_a at a full-order
% event of Algorithm 2 (a fallback or a verification), where the retained
% factorization is at hand: evaluate the exact indicators of the pair just
% screened and append eta^E/eta. Also records cal.ratio = (indicator used by
% the gate)/(exact indicator), the effectivity of the calibrated state
% indicator at the event. ra empty = self-adjoint problem (state ratio only).
% which = 'reduced' (default): the ratios of reduced pairs, cal.rs/cal.ra;
% 'corrected': the ratios of corrected pairs (compact variant), cal.rs_c/ra_c.
% The two are kept apart because a correction removes the error components
% with a large residual, so a corrected pair has a larger eta^E/eta than a
% reduced one: one median over both understates the error of reduced pairs.
if nargin < 10 || isempty(which), which = 'reduced'; end
[ex_s, ex_a] = indicator_exact(rs, ra, dJdU, Kref, Unorm);
if strcmp(which, 'corrected')
    cal.rs_c(end+1) = etaE_s/ex_s;
    if ~isempty(ra), cal.ra_c(end+1) = etaE_a/ex_a; end
else
    cal.rs(end+1) = etaE_s/ex_s;
    if ~isempty(ra), cal.ra(end+1) = etaE_a/ex_a; end
end
if isfinite(eta_s), cal.ratio(end+1) = eta_s/ex_s; end
end
