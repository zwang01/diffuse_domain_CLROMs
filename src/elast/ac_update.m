function phi = ac_update(phi, sens, freeze, phiFrozen, ac)
% AC_UPDATE  Gated Allen-Cahn design update without a volume constraint.
%
% The counterpart of ac_update_vol for problems in which the volume is
% penalized in the objective rather than constrained exactly -- as in the
% cantilever of Takezawa et al., where the compliance is minimized together
% with f times the total volume. Each of ac.Kin inner steps solves the
% semi-implicit system
%
%   (I - tau*hat_eps^2*L) phi_new = phi + tau*( -F'(phi) - beta*G'(phi).*shat )
%
% with F'(phi) = phi(1-phi)(1/2-phi) the double-well derivative, G'(phi) =
% phi^2(1-phi)^2 the gate, and shat the sensitivity normalized by its
% Euclidean norm so that beta has a scale-free meaning.
for it = 1:ac.Kin
    Fw = phi.*(1-phi).*(0.5-phi);
    G  = phi.^2.*(1-phi).^2;
    % Normalize the GATED sensitivity, as the diffuse-domain flow driver does.
    % Normalizing sens globally would divide by a norm taken over all cells,
    % while the forcing acts only where G > 0; the design term would then be
    % an order of magnitude below the double-well term and the interface would
    % not move at all.
    gs = G.*sens;
    forcing = -Fw - ac.beta*gs/(norm(gs(:)) + 1e-14);
    forcing(freeze) = 0;
    rhs = phi + ac.tau*forcing;
    if isfield(ac,'cvec'), rhs = rhs + reshape(ac.cvec, size(phi)); end
    phi = reshape(ac.Adec \ rhs(:), size(phi));
    phi = min(max(phi,0),1);
    phi(freeze) = phiFrozen(freeze);
end
end
