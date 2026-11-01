function res = tr_optimize(prob, cfg)
% TR_OPTIMIZE  The design loop of the paper on a problem adapter `prob`
% (scalar_problem, flow_problem, cantilever_problem, scalar_robin_problem): the full-order
% optimization (Algorithm 1, cfg.tol_eta = 0) or the residual-screened local
% reduced-order model (Algorithm 2, cfg.tol_eta > 0). The same code runs all
% four problems; only the adapter differs.
%
% Algorithm 2, per design iteration k:
%   * reduced state and adjoint on windowed POD bases (prob.rom_step), with
%     the Euclidean residuals and the dual-weighted estimate eta_J;
%   * screening: the calibrated indicators eta~ = eta^E/c (indicator_screen)
%     against tau_eta; c is the running median of eta^E/eta over full-order
%     events, eta the exact preconditioned indicators (indicator_exact);
%   * a rejected iteration is solved at full order WITHOUT a new factorization
%     by a Krylov method preconditioned with the retained factorization K_ref
%     (prob.pc), refactoring only when that fails or degrades, and its
%     solution enters the windows after the design update (with its predictor
%     direction when cfg.tangent);
%   * relaxation of tau_eta after N_c accepted iterations with
%     |eta_J|/|J| < tau_J/10 throughout; objective verification at full order
%     when |eta_J|/|J| exceeds theta_v*tau_J (or after n_check_max accepted
%     iterations), tightening tau_eta on a mismatch above tau_J.
%
% Options (defaults = the settings of the paper):
%  stopping  max_itr (12000), stop (true; false = exactly max_itr+1
%            iterations), win (50), sig_tol (1e-6), tol_dphi (1e-4), see stop_rule
%  screening tol_eta (1e-5; 0 = Algorithm 1), tol_eta_max (3e-5), tol_eta_min
%            (1e-9), gate ('cal' | 'eucl' | 'exact'), cal_warm (3), cal_mem
%            (10), n_warmup (2)
%            gate_mode ('state_adjoint': accept when max(eta~_s, eta~_a) <=
%            tau_eta | 'forcing': accept when eta~_s <= tau_eta and the
%            calibrated FORCING indicator eta~_g <= tol_g, see below)
%            tol_g (1e-3, scaled with tau_eta by relaxation and tightening)
%  verify    tol_J (1e-4), verify ('dwr' | 'backstop'), tol_J_dwr (5e-5),
%            n_check (25: relaxation period N_c, and the backstop period for
%            verify = 'backstop'), n_check_max (400: backstop for 'dwr'),
%            relax_mode ('consecutive': N_c accepted iterations in a row |
%            'accepted': N_c accepted iterations since the last change of
%            tau_eta, full-order events in between allowed)
%  solvers   fallback ('precond' | 'refactor'; default: the adapter's choice
%            prob.fallback for the ROM, 'refactor' for Algorithm 1), pc_tol
%            (1e-10), pc_maxit (20 per Krylov call), pc_refresh_its ([] = the
%            adapter's m_max)
%  basis     w_max (40), r_max (30), pod_tol (1e-10), lsq ('qr' | 'normal'),
%            compact (false; true: defaults to windows/ranks of eight and two
%            reduced Gauss-Newton steps; attempts up to max_corrections (2)
%            retained-factor corrections after a failed screen and up to
%            max_objective_corrections (2) after an objective trigger).
%            Corrected pairs must pass the original full-residual screen;
%            objective corrections also pass tol_J_dwr. Accepted corrected
%            pairs enter the windows. The full-order backstop is retained.
%            tangent (false; true: at every full-order event add the predictor
%            directions d = K_ref^{-1} r_s(U_k; phi_{k+1}) and
%            d_a = K_ref^{-T} r_a(W_k; U_k + d, phi_{k+1}) to the windows,
%            one linearized step toward the next design's state and adjoint),
%            tangent_tol (0: d by one application of K_ref; > 0: d by GMRES
%            on the event's Jacobian prob.jac, preconditioned with K_ref, to
%            this tolerance)
%            gate_mode 'second_chance': when the calibrated screen rejects,
%            correct the reduced pair once with the retained factorization,
%            U_c = U_r + K_ref^{-1} r_s, W_c = W_r + K_ref^{-T} r_a, estimate the
%            forcing error of the reduced pair against the corrected one, and
%            accept the CORRECTED pair if that estimate is <= tol_g (needs
%            prob.sens_of; the state gate then no longer binds)
%            gate_mode 'oracle_forcing' (measurement experiment, not a method:
%            accept on the EXACT forcing error from a full-order solve that is
%            not counted, <= tol_g, and also eta~_s <= tau_eta when
%            oracle_state = true)
%  output    errstudy (false; true: recompute every es_every-th accepted
%            iteration at full order, a measurement that changes nothing),
%            es_every (1), save_phi_at ([]), verbose (true), print_every (25)
%  experiment step_policy ('fixed'|'damped'): halve the pseudo-time step on
%            persistent opposing updates or worsening window means, down to
%            step_min_scale (1/16). Requires prob.update_scaled. This is a
%            heuristic, not a descent guarantee. stop_mode='window' requires
%            stable means and a small objective range; stop_verify=true
%            verifies a reduced stop at full order and returns that evaluated
%            design. These options leave the original defaults unchanged.
%
% Forcing indicator. The optimizer uses the design forcing f(s) (prob.forcing:
% the gated, normalized sensitivity of the Allen-Cahn step), not the adjoint
% itself. At every full-order event that follows a reduced attempt, the exact
% forcing error e_g = ||f(s_r) - f(s)|| / ||f(s)|| of the attempt is known; its
% predictor p = max(eta~_s, eta~_a) is calibrated against it,
% eta~_g = p / c_g, c_g the running median of p/e_g. Recorded in every mode;
% gates the iteration in gate_mode 'forcing' once cal_warm ratios exist.
%
% res: histories over k (J, aux (prob.aux_names), eta, eta_s, eta_a, eta_g,
% eg (exact forcing error at events), etaJ, tol, mode (1 = full-order solve,
% 0 = accepted reduced), rej (why a tried reduced pair was rejected: 1 state,
% 2 adjoint, 3 forcing), since (iterations since the last full-order event),
% r, rA, dphi, t (cumulative wall time)), counts (iters, n_fom = warm-up +
% fallbacks, n_rom, n_verify, n_verify_dwr, n_relax, n_pc, n_pc_fail,
% n_refresh, nfact), times (time, t_pc, t_fact, and T: accepted reduced
% steps, rejected attempts, event solves, window updates incl. predictors,
% calibration, design updates), designs (phi0, phi), stop_reason, cal, es, cfg.

def = struct('max_itr', 12000, 'stop', true, 'stop_consec', 2, 'win', 50, 'sig_tol', 1e-6, 'tol_dphi', 1e-4, ...
    'tol_eta', 1e-5, 'tol_eta_max', 3e-5, 'tol_eta_min', 1e-9, 'gate', 'cal', ...
    'cal_warm', 3, 'cal_mem', 10, 'n_warmup', 2, 'gate_mode', 'state_adjoint', 'tol_g', 1e-3, ...
    'tol_J', 1e-4, 'verify', 'dwr', 'tol_J_dwr', 5e-5, 'n_check', 25, 'n_check_max', 400, ...
    'relax_mode', 'consecutive', ...
    'fallback', [], 'pc_tol', 1e-10, 'pc_maxit', 20, 'pc_refresh_its', [], ...
    'w_max', 40, 'r_max', 30, 'pod_tol', 1e-10, 'lsq', 'qr', 'tangent', false, 'tangent_tol', 0, ...
    'oracle_state', true, 'compact', false, 'max_corrections', 2, 'max_objective_corrections', 2, ...
    'gn_maxit', 15, 'gn_stall', 1, 'correction_method', 'defect', 'corr_tol', 1e-3, 'corr_maxit', 8, ...
    'partial_corrections', false, 'cal_split', true, 'enrich_tol', 0, 'enrich_min_gap', 0, ...
    'errstudy', false, 'es_every', 1, 'save_phi_at', [], 'verbose', true, 'print_every', 25, ...
    'step_policy','fixed','step_min_scale',1/16,'cycle_cos',-0.25,'cycle_patience',4, ...
    'step_cooldown',50,'step_trend_tol',1e-3,'stop_mode','endpoint', ...
    'stop_spread_tol',1e-3,'stop_verify',false);
if nargin < 2, cfg = struct(); end
if isfield(cfg, 'compact') && cfg.compact
    if ~isfield(cfg,'w_max'), cfg.w_max = 8; end
    if ~isfield(cfg,'r_max'), cfg.r_max = 8; end
    if ~isfield(cfg,'gn_maxit'), cfg.gn_maxit = 2; end
end
cfg = set_defaults(cfg, def);
adaptive_step=strcmp(cfg.step_policy,'damped');
assert(ismember(cfg.step_policy,{'fixed','damped'}),'Unknown step policy');
assert(ismember(cfg.stop_mode,{'endpoint','window'}),'Unknown stopping mode');
assert(cfg.step_min_scale>0 && cfg.step_min_scale<=1);
if adaptive_step, assert(isfield(prob,'update_scaled'),'Damping needs prob.update_scaled'); end
use_rom = cfg.tol_eta > 0;
if isempty(cfg.fallback), if use_rom, cfg.fallback = prob.fallback; else, cfg.fallback = 'refactor'; end, end
if isempty(cfg.pc_refresh_its), cfg.pc_refresh_its = prob.refresh_its; end
pc_opts = struct('tol', cfg.pc_tol, 'maxit', cfg.pc_maxit, 'refresh_its', cfg.pc_refresh_its);
sa = prob.self_adjoint;
naux = numel(prob.aux_names);
relax_acc = strcmp(cfg.relax_mode, 'accepted');
gate_forcing = strcmp(cfg.gate_mode, 'forcing');
gate_oracle = strcmp(cfg.gate_mode, 'oracle_forcing');
gate_second = strcmp(cfg.gate_mode, 'second_chance');
if cfg.compact && ~strcmp(cfg.gate_mode, 'state_adjoint')
    error('Compact corrections require the standard state/adjoint residual screen.');
end
if ~ismember(cfg.correction_method, {'defect','ritz','gmres'}) || (strcmp(cfg.correction_method,'ritz') && ~sa)
    error('Ritz correction is available only for a self-adjoint problem.');
end

ddm_count_fact(0);
tol_eta = cfg.tol_eta;
rom = prob.rom_new(struct('w_max', cfg.w_max, 'r_max', cfg.r_max, 'pod_tol', cfg.pod_tol, 'lsq', cfg.lsq, 'gn_maxit', cfg.gn_maxit, 'gn_stall', cfg.gn_stall));
rom.correction_method = cfg.correction_method;
rom.corr_tol = cfg.corr_tol; rom.corr_maxit = cfg.corr_maxit;   % 'gmres' corrections: inexact Newton step through K_ref
cal = struct('rs', [], 'ra', [], 'ratio', [], 'rg', [], 'warm', cfg.cal_warm, 'mem', cfg.cal_mem, ...
             'rs_c', [], 'ra_c', [], 'split', cfg.cal_split);   % rs_c, ra_c: ratios of corrected pairs
Kref = []; Unorm = NaN;                      % retained factorization and ||U_ref||
if strcmp(cfg.verify, 'dwr'), n_check_cur = cfg.n_check_max; else, n_check_cur = cfg.n_check; end
n_fom = 0; n_rom = 0; n_verify = 0; n_verify_dwr = 0; n_relax = 0;
n_pc = 0; n_pc_fail = 0; n_refresh = 0; pc_its = []; t_pc = 0; t_fact = 0;
n_since_check = 0; n_since_relax = 0; etaJ_max = 0; since = 0; n_second = 0;
TT = struct('rom_acc', 0, 'rom_rej', 0, 'solve', 0, 'enrich', 0, 'cal', 0, 'update', 0, 'second', 0, 'correct', 0);
n_correct_screen = 0; n_correct_objective = 0;
n_correct_screen_attempt = 0; n_correct_objective_attempt = 0;
nan1 = nan(cfg.max_itr+1, 1);
H = struct('J', nan1, 'aux', nan(cfg.max_itr+1, naux), 'eta', nan1, 'eta_s', nan1, 'eta_a', nan1, ...
           'eta_g', nan1, 'eg', nan1, 'etaJ', nan1, 'tol', nan1, 'mode', nan1, 'rej', zeros(cfg.max_itr+1,1), ...
           'since', nan1, 'r', nan1, 'rA', nan1, 'dphi', nan1, 't', nan1, ...
           'source', nan1, 'correction_steps', zeros(cfg.max_itr+1,1), ...
           'step_scale',ones(cfg.max_itr+1,1),'step_reason',zeros(cfg.max_itr+1,1));
es = []; phi_saved = struct('k', {}, 'phi', {});
phi = prob.phi0; Uwarm = []; stop_reason = 'iteration cap';
last_enrich=-Inf; n_enrich=0; n_enrich_skipped=0;
step_scale=1; previous_step=[]; cycle_count=0; last_damping=-Inf;
n_stop_verify=0; terminal_verified=false;
tStart = tic;

for k = 0:cfg.max_itr
    if any(k == cfg.save_phi_at)
        saved=phi; if isfield(prob,'export_design'), saved=prob.export_design(phi); end
        phi_saved(end+1)=struct('k',k,'phi',saved); %#ok<AGROW>
    end
    pending = [];                                % full-order pair to enter the windows after the update
    corrected = false; verified = false;

    % ---- reduced state and adjoint, screening
    accepted = false; tried = false; eta_s = NaN; eta_a = NaN; eta_g = NaN; etaJ = NaN; R = []; rej = 0;
    if use_rom && k >= cfg.n_warmup && rom.r > 0 && ~isempty(Uwarm)
        t0 = tic;
        R = prob.rom_step(phi, rom, Uwarm);
        t_step = toc(t0);
        tried = true;
        if isfield(R, 'rom') && ~isempty(R.rom), rom = R.rom; R.rom = []; end   % rebuilt (band-limited step)
        if ~isempty(Kref) && (strcmp(cfg.gate, 'exact') || (strcmp(cfg.gate, 'cal') && numel(cal.rs) < cal.warm))
            R = full_res(R);                     % the exact indicators need the residual vectors
        end
        t0 = tic;
        [eta_s, eta_a, cal] = indicator_screen(cal, cfg.gate, R.etaE_s, R.etaE_a, R.rs, R.ra, R.dJdU, Kref, Unorm, sa);
        TT.cal = TT.cal + toc(t0);
        pred = max(eta_s, eta_a);
        if numel(cal.rg) >= cal.warm, eta_g = pred/median(cal.rg(max(1,end-cal.mem+1):end)); end
        if gate_oracle
            % measurement-mode gate: the exact forcing error from a full-order solve that is
            % not part of the method (factorization count restored)
            nf = ddm_count_fact([]);
            outO = prob.fom(phi, Uwarm, false);
            ddm_count_fact(nf);
            eta_g = forcing_err(prob, phi, R.sens, outO.sens); H.eg(k+1) = eta_g;
            tol_g = cfg.tol_g*(tol_eta/cfg.tol_eta);
            accepted = eta_g <= tol_g && (~cfg.oracle_state || eta_s <= tol_eta);
            if ~accepted, rej = 3 - 2*(eta_g <= tol_g); end             % 1 state, 3 forcing
        elseif gate_forcing && ~isnan(eta_g)
            tol_g = cfg.tol_g*(tol_eta/cfg.tol_eta);
            accepted = eta_s <= tol_eta && eta_g <= tol_g;
            if ~accepted, rej = 1 + 2*(eta_s <= tol_eta); end           % 1 state, 3 forcing
        else
            accepted = max(eta_s, eta_a) <= tol_eta;
            if ~accepted, rej = 1 + (eta_s <= tol_eta); end             % 1 state, 2 adjoint
        end
        etaJ = R.etaJ; R0 = []; nc = 0;
        if ~accepted && cfg.compact && cfg.max_corrections > 0 && ~isempty(Kref)
            t0 = tic;
            if cfg.cal_split, R0 = full_res(R); end   % the reduced pair, calibrated separately at the fallback
            [R, accepted, eta_s, eta_a, nc] = try_corrections(prob, phi, rom, R, Kref, cal, ...
                cfg, Unorm, tol_eta, cfg.max_corrections, false, eta_s, eta_a);
            TT.correct = TT.correct + toc(t0);
            n_correct_screen_attempt = n_correct_screen_attempt + (nc > 0);
            H.correction_steps(k+1) = H.correction_steps(k+1) + nc;
            etaJ = R.etaJ; pred = max(eta_s, eta_a);
            if accepted
                corrected = true; rej = 0; n_correct_screen = n_correct_screen + 1;
                pending = struct('U', R.U, 'W', R.W);
                if isfield(R, 'cmode') && strcmp(R.cmode, 'a'), pending.U = []; end   % reduced state: already in the window's span
            end
        end
        if ~accepted && gate_second && ~isempty(Kref)
            % second chance: one correction step with the retained factorization, the
            % forcing error of the reduced pair estimated against the corrected pair,
            % and the corrected pair accepted when that estimate is below tol_g
            t0 = tic;
            R = full_res(R);
            Uc = R.U + Kref.solve(R.rs);
            if sa, Wc = []; else, Wc = R.W + Kref.solvet(R.ra); end
            [sc, Jc, auxc] = prob.sens_of(Uc, Wc, phi);
            eta_g = forcing_err(prob, phi, R.sens, sc);
            tol_g = cfg.tol_g*(tol_eta/cfg.tol_eta);
            if eta_g <= tol_g
                accepted = true; rej = 0; n_second = n_second + 1;
                R.U = Uc; R.W = Wc; R.sens = sc; R.J = Jc; R.aux = auxc;
            end
            TT.second = TT.second + toc(t0);
        end
        if accepted, TT.rom_acc = TT.rom_acc + t_step; else, TT.rom_rej = TT.rom_rej + t_step; end
        if ~accepted && strcmp(cfg.gate,'cal') && ~isempty(Kref) && numel(cal.rs) >= cal.warm
            t0 = tic;
            R = full_res(R);
            if cfg.cal_split && nc > 0 && ~isempty(R0)
                cal = indicator_calibrate(cal, R0.etaE_s, R0.etaE_a, R0.rs, R0.ra, R0.dJdU, Kref, Unorm, NaN);
                cal = indicator_calibrate(cal, R.etaE_s, R.etaE_a, R.rs, R.ra, R.dJdU, Kref, Unorm, eta_s, 'corrected');
            else
                cal = indicator_calibrate(cal, R.etaE_s, R.etaE_a, R.rs, R.ra, R.dJdU, Kref, Unorm, eta_s);
            end
            TT.cal = TT.cal + toc(t0);
        end
    end

    if ~accepted
        % ---- full-order solve: warm-up, fallback, or Algorithm 1
        if tried, R=full_res(R); end
        t0 = tic;
        pc_ok = false;
        if strcmp(cfg.fallback, 'precond') && ~isempty(Kref)
            if tried, U0 = R.U; W0 = R.W; else, U0 = Uwarm; W0 = []; end
            t1 = tic;
            [outp, pinfo] = prob.pc(phi, U0, W0, Kref, pc_opts);
            t_pc = t_pc + toc(t1);
            pc_ok = pinfo.ok;
            if pc_ok
                out = outp; n_pc = n_pc + 1; pc_its(end+1) = sum(pinfo.its); %#ok<AGROW>
                if pinfo.refreshed, n_refresh = n_refresh + 1; end
            else
                n_pc_fail = n_pc_fail + 1;
            end
        end
        if ~pc_ok                               % factor
            t1 = tic;
            out = prob.fom(phi, Uwarm, use_rom);
            t_fact = t_fact + toc(t1);
        end
        TT.solve = TT.solve + toc(t0);
        U = out.U; W = out.W; J = out.J; sens = out.sens; aux = out.aux;
        Kref = out.Kref; Unorm = norm(U);
        if tried, [cal, H.eg(k+1)] = forcing_cal(cal, prob, phi, R.sens, sens, pred); end
        if use_rom, pending = struct('U', U, 'W', W); end
        Uwarm = U;
        n_fom = n_fom + 1; n_since_check = 0; since = 0;
        if ~relax_acc, n_since_relax = 0; etaJ_max = 0; end
    else
        U = R.U; W = R.W; J = R.J; sens = R.sens; aux = R.aux;
        n_rom = n_rom + 1; n_since_check = n_since_check + 1; n_since_relax = n_since_relax + 1;
        since = since + 1;
        etaJ_max = max(etaJ_max, abs(etaJ)/max(abs(J), 1e-30));
        if ~isempty(U), Uwarm = U; end           % a lazy scalar pair needs no state warm start
    end

    % ---- error study (measurement only): full-order solve at the same design
    if cfg.errstudy && accepted && mod(n_rom, cfg.es_every) == 0
        R=full_res(R); U=R.U; W=R.W; sens=R.sens;
        nf = ddm_count_fact([]);
        outF = prob.fom(phi, Uwarm, false);     % reference adjoint factored at the converged state
        ddm_count_fact(nf);                     % not part of the method's cost
        R = full_res(R);
        [ePs, ePa] = indicator_exact(R.rs, R.ra, R.dJdU, Kref, Unorm);
        e = struct('k', k, 'etaE_s', R.etaE_s, 'etaE_a', R.etaE_a, 'etaP_s', ePs, 'etaP_a', ePa, ...
            'eta_s', eta_s, 'eta_a', eta_a, 'eta_g', eta_g, 'etaJ', etaJ, ...
            'err_s', norm(outF.U - U)/norm(outF.U), 'J', J, 'JF', outF.J, ...
            'err_g', forcing_err(prob, phi, sens, outF.sens));
        if sa, e.err_a = e.err_s; else, e.err_a = norm(outF.W - W)/norm(outF.W); end
        % errors on the physically determined components only (prob.err_mask: the
        % flow's velocities; its pressure in the solid is fixed only by the tiny
        % regularization and dominates the full-vector error at no cost to anything else)
        if isfield(prob, 'err_mask') && ~isempty(prob.err_mask)
            m = prob.err_mask;
            e.err_s_m = norm(outF.U(m) - U(m))/norm(outF.U(m));
            if sa, e.err_a_m = e.err_s_m; else, e.err_a_m = norm(outF.W(m) - W(m))/norm(outF.W(m)); end
        else
            e.err_s_m = e.err_s; e.err_a_m = e.err_a;
        end
        es = [es, e]; %#ok<AGROW>
    end

    % ---- relaxation of tau_eta: N_c accepted iterations with |eta_J|/|J| < tau_J/10 throughout
    if accepted && n_since_relax >= cfg.n_check
        if etaJ_max < cfg.tol_J/10, tol_eta = min(2*tol_eta, cfg.tol_eta_max); n_relax = n_relax + 1; end
        n_since_relax = 0; etaJ_max = 0;
    end
    % ---- objective verification
    dwr_trig = strcmp(cfg.verify,'dwr') && isfinite(etaJ) && abs(etaJ)/max(abs(J),1e-30) > cfg.tol_J_dwr;
    if accepted && dwr_trig && n_since_check < n_check_cur && cfg.compact && cfg.max_objective_corrections > 0
        t0 = tic;
        [Rc, okc, esc, eac, nc] = try_corrections(prob, phi, rom, R, Kref, cal, ...
            cfg, Unorm, tol_eta, cfg.max_objective_corrections, true);
        TT.correct = TT.correct + toc(t0);
        n_correct_objective_attempt = n_correct_objective_attempt + (nc > 0);
        H.correction_steps(k+1) = H.correction_steps(k+1) + nc;
        if okc
            R = Rc; eta_s = esc; eta_a = eac; etaJ = R.etaJ;
            U = R.U; W = R.W; J = R.J; sens = R.sens; aux = R.aux; Uwarm = U;
            pred = max(eta_s, eta_a); corrected = true;
            pending = struct('U', U, 'W', W);
            n_correct_objective = n_correct_objective + 1;
            dwr_trig = false;
        end
    end
    if accepted && (n_since_check >= n_check_cur || dwr_trig)
        verified = true;
        R=full_res(R); U=R.U; W=R.W; sens=R.sens;
        if strcmp(cfg.gate,'cal') && ~isempty(Kref)
            t0 = tic;
            R = full_res(R);
            if cfg.cal_split && corrected, which = 'corrected'; else, which = 'reduced'; end
            cal = indicator_calibrate(cal, R.etaE_s, R.etaE_a, R.rs, R.ra, R.dJdU, Kref, Unorm, eta_s, which);
            TT.cal = TT.cal + toc(t0);
        end
        t0 = tic;
        pc_ok = false;
        if strcmp(cfg.fallback, 'precond') && ~isempty(Kref)
            t1 = tic;
            [outF, pinfo] = prob.pc(phi, U, W, Kref, pc_opts);
            t_pc = t_pc + toc(t1);
            pc_ok = pinfo.ok;
            if pc_ok
                n_pc = n_pc + 1; pc_its(end+1) = sum(pinfo.its); %#ok<AGROW>
                if pinfo.refreshed, n_refresh = n_refresh + 1; end
            else
                n_pc_fail = n_pc_fail + 1;
            end
        end
        if ~pc_ok
            t1 = tic;
            outF = prob.fom(phi, Uwarm, true);
            t_fact = t_fact + toc(t1);
        end
        TT.solve = TT.solve + toc(t0);
        relJ = abs(J - outF.J)/max(abs(outF.J), 1e-14);
        if relJ > cfg.tol_J                     % objective not tracked: tighten
            tol_eta = max(tol_eta/2, cfg.tol_eta_min);
            if cfg.verbose, fprintf('[verify] k = %d: objective mismatch %.2e, tol_eta -> %.2e\n', k, relJ, tol_eta); end
        elseif relJ < cfg.tol_J/10              % room to spare: relax
            tol_eta = min(2*tol_eta, cfg.tol_eta_max);
        end
        if dwr_trig, n_verify_dwr = n_verify_dwr + 1; end
        [cal, H.eg(k+1)] = forcing_cal(cal, prob, phi, sens, outF.sens, pred);
        Kref = outF.Kref; Unorm = norm(outF.U);
        pending = struct('U', outF.U, 'W', outF.W);
        % continue from the verified full-order pair
        U = outF.U; W = outF.W; J = outF.J; sens = outF.sens; aux = outF.aux; Uwarm = U;
        n_verify = n_verify + 1; n_since_check = 0; since = 0;
        n_since_relax = 0; etaJ_max = 0;
    end

    % ---- histories
    H.J(k+1) = J; H.aux(k+1,:) = aux; H.eta(k+1) = max(eta_s, eta_a);
    H.eta_s(k+1) = eta_s; H.eta_a(k+1) = eta_a; H.eta_g(k+1) = eta_g; H.etaJ(k+1) = etaJ;
    H.tol(k+1) = tol_eta; H.mode(k+1) = double(~accepted); H.rej(k+1) = rej; H.since(k+1) = since;
    H.r(k+1) = rom.r; H.rA(k+1) = rom.rA;
    H.source(k+1) = double(~accepted) + 2*double(corrected && accepted);
    if verified, H.source(k+1) = 3; end

    % ---- design update
    t0 = tic;
    if adaptive_step
        phinew=prob.update_scaled(phi,sens,step_scale);
        delta=phinew(:)-phi(:);
        if ~isempty(previous_step)
            cosine=(previous_step.'*delta)/max(norm(previous_step)*norm(delta),1e-300);
            if cosine<cfg.cycle_cos, cycle_count=cycle_count+1; else, cycle_count=0; end
        end
        worsening=false;
        if k+1>=2*cfg.win && mod(k+1,cfg.win)==0
            recent=mean(H.J(k+2-cfg.win:k+1)); older=mean(H.J(k+2-2*cfg.win:k+1-cfg.win));
            worsening=recent-older>cfg.step_trend_tol*max(abs(older),1e-30);
        end
        if (cycle_count>=cfg.cycle_patience || worsening) && ...
                k-last_damping>=cfg.step_cooldown && step_scale>cfg.step_min_scale
            step_scale=max(step_scale/2,cfg.step_min_scale);
            phinew=prob.update_scaled(phi,sens,step_scale);
            H.step_reason(k+1)=1+double(worsening); last_damping=k; cycle_count=0;
        end
        previous_step=phinew(:)-phi(:);
    else
        phinew = prob.update(phi, sens);
    end
    H.step_scale(k+1)=step_scale;
    TT.update = TT.update + toc(t0);
    if isfield(prob,'design_change'), dphi=prob.design_change(phi,phinew);
    else, dphi = max(abs(phinew(:) - phi(:))); end

    % Experimental stopping: check the evaluated design, and verify a ROM
    % stop with a full-order pair. Damping alone must not cause a small-step
    % stop, so the design-change safeguard is normalized by the step scale.
    H.dphi(k+1)=dphi;
    if adaptive_step, stop_dphi=H.dphi(1:k+1)./H.step_scale(1:k+1);
    else, stop_dphi=H.dphi; end
    [stop,why]=stop_rule(cfg,k,H.J,stop_dphi);
    if stop && cfg.stop_verify && accepted && ~verified
        t0=tic; outF=prob.fom(phi,Uwarm,true); elapsed=toc(t0);
        terminal_mismatch=abs(J-outF.J)/max(abs(outF.J),1e-30);
        TT.solve=TT.solve+elapsed; t_fact=t_fact+elapsed;
        n_verify=n_verify+1; n_stop_verify=n_stop_verify+1; verified=true;
        Kref=outF.Kref; Unorm=norm(outF.U); Uwarm=outF.U;
        pending=struct('U',outF.U,'W',outF.W);
        J=outF.J; sens=outF.sens; H.J(k+1)=J; H.aux(k+1,:)=outF.aux;
        H.source(k+1)=3; H.since(k+1)=0; n_since_check=0; since=0; n_since_relax=0; etaJ_max=0;
        t0=tic;
        if adaptive_step, phinew=prob.update_scaled(phi,sens,step_scale);
        else, phinew=prob.update(phi,sens); end
        TT.update=TT.update+toc(t0);
        if isfield(prob,'design_change'), dphi=prob.design_change(phi,phinew);
        else, dphi=max(abs(phinew(:)-phi(:))); end
        H.dphi(k+1)=dphi;
        stop_dphi(k+1)=dphi/step_scale;
        [stop,why]=stop_rule(cfg,k,H.J,stop_dphi);
        if terminal_mismatch>cfg.tol_J
            stop=false; tol_eta=max(tol_eta/2,cfg.tol_eta_min);
        end
        if adaptive_step, previous_step=phinew(:)-phi(:); end
    end
    if stop && cfg.stop_verify, terminal_verified=true; end

    % ---- window update with the full-order pair of this iteration (and its predictors)
    if ~isempty(pending)
        t0 = tic;
        skip=false;
        if rom.r>=cfg.r_max && (sa || rom.rA>=cfg.r_max)
            skip=k-last_enrich<cfg.enrich_min_gap;
            if corrected && ~verified && cfg.enrich_tol>0
                nov=0;
                if ~isempty(pending.U), nov=norm(pending.U-rom.Phi*(rom.Phi.'*pending.U))/max(norm(pending.U),1e-30); end
                if ~sa, nov=max(nov,norm(pending.W-rom.PhiA*(rom.PhiA.'*pending.W))/max(norm(pending.W),1e-30)); end
                skip=skip || nov<cfg.enrich_tol;
            end
        end
        if skip
            n_enrich_skipped=n_enrich_skipped+1;
        else
        Us = pending.U; Ws = pending.W;
        if cfg.tangent
            rs = prob.res_s(pending.U, phinew);
            if cfg.tangent_tol > 0                % linearized step solved by GMRES on the event's Jacobian
                Jk = prob.jac(pending.U, phi);
                d = pc_gmres(Jk, rs, [], Kref.solve, cfg.tangent_tol, cfg.pc_maxit);
            else                                  % one application of the retained factorization
                d = Kref.solve(rs);
            end
            Us = [Us, d];
            if ~sa
                ra = prob.res_a(pending.W, pending.U + d, phinew);
                if cfg.tangent_tol > 0
                    da = pc_gmres(Jk.', ra, [], Kref.solvet, cfg.tangent_tol, cfg.pc_maxit);
                else
                    da = Kref.solvet(ra);
                end
                Ws = [Ws, da];
            end
        end
        rom = prob.enrich(rom, Us, Ws, phinew);
        last_enrich=k; n_enrich=n_enrich+1;
        end
        TT.enrich = TT.enrich + toc(t0);
    end

    H.dphi(k+1) = dphi; H.t(k+1) = toc(tStart);
    if cfg.verbose && mod(k, cfg.print_every) == 0
        if accepted, lbl = 'ROM'; else, lbl = 'FOM'; end
        fprintf('%s k = %5d  J = %.7e  [%s] eta = %8.2e  r = %2d/%2d  |dphi| = %.2e  fact = %d\n', ...
            prob.name, k, J, lbl, max(eta_s, eta_a), rom.r, rom.rA, dphi, ddm_count_fact([]));
    end
    if ~(stop && cfg.stop_verify), phi = phinew; end
    if stop, stop_reason = why; break, end
end

phi0=prob.phi0;
if isfield(prob,'export_design'), phi=prob.export_design(phi); phi0=prob.export_design(phi0); end
res = struct('name', prob.name, 'cfg', cfg, 'iters', k+1, 'k_last', k, 'stop_reason', stop_reason, ...
    'n_enrich', n_enrich, 'n_enrich_skipped', n_enrich_skipped, ...
    'n_stop_verify',n_stop_verify,'terminal_verified',terminal_verified, ...
    'n_fom', n_fom, 'n_rom', n_rom, 'n_verify', n_verify, 'n_verify_dwr', n_verify_dwr, 'n_relax', n_relax, ...
    'n_pc', n_pc, 'n_pc_fail', n_pc_fail, 'n_refresh', n_refresh, 'pc_its', pc_its, 'n_second', n_second, ...
    'n_correct_screen', n_correct_screen, 'n_correct_objective', n_correct_objective, ...
    'n_correct_screen_attempt', n_correct_screen_attempt, 'n_correct_objective_attempt', n_correct_objective_attempt, ...
    'nfact', ddm_count_fact([]), 'time', toc(tStart), 't_pc', t_pc, 't_fact', t_fact, 'T', TT, ...
    'phi0', phi0, 'phi', phi, 'cal', cal, 'es', es);
res.aux_names = prob.aux_names;
res.phi_saved = phi_saved;
fn = fieldnames(H);
for i = 1:numel(fn), res.(fn{i}) = H.(fn{i})(1:k+1, :); end
if isfield(prob, 'info'), res.info = prob.info; end
if isfield(prob,'stats'), res.problem_stats=prob.stats(); end
if cfg.verbose
    fprintf(['[%s] %s after %d iterations, %.1f s: full-order %d (+%d verify), reduced %d, ' ...
             'factorizations %d, GMRES/CG solves %d (%d failed), J = %.7e\n'], prob.name, stop_reason, ...
        res.iters, res.time, n_fom, n_verify, n_rom, res.nfact, n_pc, n_pc_fail, J);
end
end

% =====================================================================
function [R, ok, es, ea, count] = try_corrections(prob, phi, rom, R, Kref, cal, cfg, Unorm, tol_eta, limit, objective, eta_s0, eta_a0)
% Freeze calibration while trying corrections. Each candidate is screened on
% its freshly evaluated full residuals, with the same tolerances as the ROM.
% A pair whose state passed the screen is corrected in the adjoint only
% (mode 'a': one triangular solve, the state and its window untouched);
% otherwise both are corrected ('sa'). Objective corrections correct both.
R = full_res(R); ok = false; count = 0; es = Inf; ea = Inf;
if isfield(cal, 'split') && cal.split && numel(cal.rs_c) >= cal.warm      % corrected pairs: their own calibration
    cal.rs = cal.rs_c; if ~isempty(cal.ra_c), cal.ra = cal.ra_c; end
end
if nargin < 13, eta_s0 = Inf; eta_a0 = Inf; end
if objective || prob.self_adjoint || eta_s0 > tol_eta || ~cfg.partial_corrections, mode = 'sa'; else, mode = 'a'; end
es = eta_s0; ea = eta_a0;
for j = 1:limit
    % residual reduction the screen requires of this attempt ('gmres' corrections stop there)
    if objective, tgt = 0.5*cfg.tol_J_dwr*max(abs(R.J),1e-30)/max(abs(R.etaJ),1e-300); R.corr_target_s = min(tgt, 0.5); R.corr_target_a = min(tgt, 0.5);
    else, R.corr_target_s = min(0.5*tol_eta/max(es,1e-300), 0.5); R.corr_target_a = min(0.5*tol_eta/max(ea,1e-300), 0.5); end
    C = prob.correct(phi, rom, R, Kref, mode);
    C.cmode = mode;
    count = j;
    if any(~isfinite(C.U)) || any(~isfinite(C.W)) || ~isfinite(C.J), break, end
    R = C;
    [es, ea] = indicator_screen(cal, cfg.gate, R.etaE_s, R.etaE_a, ...
        R.rs, R.ra, R.dJdU, Kref, Unorm, prob.self_adjoint);
    ok = max(es, ea) <= tol_eta && isfinite(R.etaJ);
    if objective, ok = ok && abs(R.etaJ)/max(abs(R.J),1e-30) <= cfg.tol_J_dwr; end
    if ok, return, end
    if ~objective && ~prob.self_adjoint && es <= tol_eta && cfg.partial_corrections, mode = 'a'; else, mode = 'sa'; end
end
end

function R = full_res(R)
if isfield(R,'materialize') && ~isempty(R.materialize), R=R.materialize(); end
% the full-length residual vectors of a reduced pair, formed on demand when the
% reduced step returned them only implicitly (band-limited step, R.res_fun)
if isempty(R.rs) && isfield(R, 'res_fun') && ~isempty(R.res_fun)
    F = R.res_fun(); R.rs = F.rs; R.ra = F.ra; R.dJdU = F.dJdU;
end
end

function e = forcing_err(prob, phi, s_r, s)
f = prob.forcing(phi, s);
e = norm(f(:) - reshape(prob.forcing(phi, s_r), [], 1))/max(norm(f(:)), 1e-300);
end

function [cal, e] = forcing_cal(cal, prob, phi, s_r, s, pred)
% exact forcing error of the reduced pair at a full-order event, and the
% calibration ratio of its predictor
e = forcing_err(prob, phi, s_r, s);
if e > 1e-14 && isfinite(pred), cal.rg(end+1) = pred/e; end
end
