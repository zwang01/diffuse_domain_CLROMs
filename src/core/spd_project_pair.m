function [C, As, Aa] = spd_project_pair(C, P, Q, GP, GQ, w, d)
% Exact coefficient updates of two Galerkin matrices. Global scans find
% changes; only changed rows contribute to each update. Basis changes must
% clear C. A periodic complete assembly controls accumulated roundoff.
rebuild=isempty(C);
if ~rebuild
    iw=find(w~=C.w); id=find(d~=C.d);
    rebuild=C.age>=100 || numel(iw)+numel(id)>.5*(numel(w)+numel(d));
end
if rebuild
    As=GP.'*(w.*GP)+P.'*(d.*P);
    Aa=GQ.'*(w.*GQ)+Q.'*(d.*Q);
    age=0;
else
    dw=w(iw)-C.w(iw); dd=d(id)-C.d(id);
    As=C.As+GP(iw,:).'*(dw.*GP(iw,:))+P(id,:).'*(dd.*P(id,:));
    Aa=C.Aa+GQ(iw,:).'*(dw.*GQ(iw,:))+Q(id,:).'*(dd.*Q(id,:));
    age=C.age+1;
end
As=(As+As.')/2; Aa=(Aa+Aa.')/2;
C=struct('w',w,'d',d,'As',As,'Aa',Aa,'age',age);
end
