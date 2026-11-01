function V = compact_dwr_basis(Phi, corrected, residual)
% COMPACT_DWR_BASIS  Adjoint space for a corrected (generally non-POD) pair.
% Keep the accepted adjoint direction as well as the residual enrichment.
V = Phi;
for z = [corrected, residual]
    nz = norm(z);
    if nz == 0, continue, end
    q = z - V*(V.'*z);
    q = q - V*(V.'*q);
    nq = norm(q);
    if nq > 1e-10*nz, V = [V, q/nq]; end %#ok<AGROW>
end
end
