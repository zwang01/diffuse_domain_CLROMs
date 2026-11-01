function [stop, reason] = stop_rule(cfg, k, Jhist, dphihist)
% STOP_RULE  Stopping rule of the paper (Remark rem:stopping):
%   objective stagnation  |J_k - J_{k-w}| <= w*sigma*|J_k|  (w = cfg.win,
%                         sigma = cfg.sig_tol), the test that decides
%                         convergence, and
%   design safeguard      max |phi_{j+1} - phi_j|_inf over the last w steps
%                         below cfg.tol_dphi (0 = off), against a frozen design.
% The design test cannot be primary: the Allen-Cahn forcing uses the
% normalized sensitivity, so the step does not tend to zero.
% The stagnation test must hold at cfg.stop_consec consecutive iterations
% (default 2): a single upward jump of the objective after a steady descent
% can cancel the difference over one window and would otherwise stop a run
% that is still descending (seen on the fine flow grid at tau = 0.01).
% cfg.stop = false switches the rule off (fixed number of iterations).
stop = false; reason = '';
if ~cfg.stop || k < cfg.win, return, end
if isfield(cfg, 'stop_consec'), nc = cfg.stop_consec; else, nc = 2; end
if k < cfg.win + nc - 1, return, end
if isfield(cfg,'stop_mode') && strcmp(cfg.stop_mode,'window')
    % A plateau needs both stable window means and small within-window
    % variation; matching endpoints of an oscillation is insufficient.
    if k+1<2*cfg.win+nc-1, return,end
    ok=true;
    for j=0:nc-1
        v=Jhist(k+2-cfg.win-j:k+1-j);
        u=Jhist(k+2-2*cfg.win-j:k+1-cfg.win-j);
        scale=max(abs(mean(v)),1e-30);
        ok=ok && abs(mean(v)-mean(u))<=cfg.win*cfg.sig_tol*scale ...
            && max(v)-min(v)<=cfg.stop_spread_tol*scale;
    end
    if ok, stop=true; reason=sprintf('stable objective windows (%d iterations each)',cfg.win); end
    return
end
ok = true;
for j = 0:nc-1
    J = Jhist(k+1-j);
    ok = ok && abs(J - Jhist(k+1-j-cfg.win)) <= cfg.win*cfg.sig_tol*abs(J);
end
if ok
    stop = true; reason = sprintf('objective stagnation over %d iterations', cfg.win); return
end
J = Jhist(k+1);
if cfg.tol_dphi > 0 && max(dphihist(k+2-cfg.win:k+1)) < cfg.tol_dphi
    stop = true; reason = sprintf('design stagnant (max|dphi| < %.0e over %d iterations)', cfg.tol_dphi, cfg.win);
end
end
