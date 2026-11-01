function [eta_s, eta_a, cal] = indicator_screen(cal, gate, etaE_s, etaE_a, rs, ra, dJdU, Kref, Unorm, sa)
% INDICATOR_SCREEN  The indicators that the acceptance test of Algorithm 2
% compares with tau_eta.
%   gate = 'cal'  (the method): calibrated indicators eta~ = eta^E / c, with
%                 c_s, c_a the medians of eta^E/eta over the last cal.mem
%                 direct evaluations of the exact indicators (indicator_exact).
%                 The first cal.warm reduced iterations after the warm-up use
%                 the exact indicators and seed the ratios.
%   gate = 'eucl' : the uncalibrated Euclidean residuals (ablation).
%   gate = 'exact': the exact preconditioned indicators at every iteration
%                 (ablation; costs triangular solves at every iteration).
% etaE_s, etaE_a are the Euclidean indicators of the reduced step; cal holds
% the ratio histories cal.rs, cal.ra and the constants cal.warm, cal.mem.
% For a self-adjoint problem (sa = true) eta_a = 0. rs, ra, dJdU are used only
% while the exact indicators are evaluated (gate 'exact' or calibration warm-up);
% a band-limited step may pass them empty otherwise.
eta_s = etaE_s; eta_a = etaE_a;
if strcmp(gate, 'eucl') || isempty(Kref), return, end
if nargin < 10, sa = isempty(ra); end
if strcmp(gate, 'exact') || numel(cal.rs) < cal.warm
    [eta_s, eta_a] = indicator_exact(rs, ra, dJdU, Kref, Unorm);
    if strcmp(gate, 'cal')
        cal.rs(end+1) = etaE_s/eta_s;
        if ~sa, cal.ra(end+1) = etaE_a/eta_a; end
    end
else
    eta_s = etaE_s / median(cal.rs(max(1,end-cal.mem+1):end));
    if sa, eta_a = 0;
    else,  eta_a = etaE_a / median(cal.ra(max(1,end-cal.mem+1):end));
    end
end
end
