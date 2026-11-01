function n = ddm_count_fact(reset)
% DDM_COUNT_FACT  Counter of sparse factorizations of the full-order operator,
% the machine-independent cost measure reported in the tables (a reduced
% iteration never factors).
%   ddm_count_fact        count one factorization (called before each one)
%   ddm_count_fact(0)     reset the counter
%   n = ddm_count_fact    count one and return the total
%   n = ddm_count_fact([]) return the total without counting
global DDM_NFACT
if isempty(DDM_NFACT), DDM_NFACT = 0; end
if nargin == 0
    DDM_NFACT = DDM_NFACT + 1;
elseif ~isempty(reset)
    DDM_NFACT = reset;
end
n = DDM_NFACT;
end
