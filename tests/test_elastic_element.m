function test_elastic_element()
p=cantilever_problem(struct('Nx',40,'Ny',20));
q=cantilever_problem(struct('Nx',40,'Ny',20,'element_ops',true));
o=struct('w_max',8,'r_max',8,'pod_tol',1e-10,'lsq','qr');
rp=p.rom_new(o); rq=q.rom_new(o); phi=p.phi0;
for k=1:8
    f=p.fom(phi,[],false); rp=p.enrich(rp,f.U,[],phi); rq=q.enrich(rq,f.U,[],phi);
    phi=p.update(phi,f.sens);
end
a=p.rom_step(phi,rp,f.U); b=q.rom_step(phi,rq,f.U);
for k=0:2
    assert(norm(a.U-b.U)/norm(a.U)<2e-8);
    assert(norm(a.sens-b.sens,'fro')/norm(a.sens,'fro')<2e-7);
    assert(norm(a.rs-b.rs)<1e-8);
    assert(abs(a.etaJ-b.etaJ)<2e-8*abs(a.J));
    a=p.correct(phi,rp,a,f.Kref); b=q.correct(phi,rq,b,f.Kref);
end
fprintf('Element projection/operator checks passed\n');
end
