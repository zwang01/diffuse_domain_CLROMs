function eta = spd_dwr_extend(P, Ar, E, Afun, g, rs)
% Assemble only the enrichment blocks of a symmetric Galerkin adjoint.
if isempty(E), eta=(Ar\(P.'*g)).'*(P.'*rs); return, end
AE=Afun(E); B=P.'*AE;
M=[Ar,B;B.',E.'*AE]; M=(M+M.')/2;
c=M\[P.'*g;E.'*g];
eta=c.'*[P.'*rs;E.'*rs];
end
