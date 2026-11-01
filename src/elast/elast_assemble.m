function K = elast_assemble(S, kphi)
% ELAST_ASSEMBLE  K(phi) = sum_e k(phi_e) Ke, from the precomputed pattern.
% Affine in the element weights kphi, which is what lets the reduced operator
% be formed exactly for any design.
sK = S.Kevec * kphi(:).';              % 64 x nel
K  = sparse(S.iK, S.jK, sK(:), S.ndof, S.ndof);
K  = (K + K.')/2;                      % symmetrize against round-off
end
