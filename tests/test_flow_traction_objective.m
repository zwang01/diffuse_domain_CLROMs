function report = test_flow_traction_objective()
% Check the complete boundary-momentum objective and its discrete adjoint.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'src')));
p=balanced_ddm_problem('flow',struct('case_options',struct( ...
    'Nx',80,'penalty_C',64,'penalty_power',2,'drag_form','traction')));
phi=p.phi0; a=p.fom(phi,[],false);
par=p.info.par; S=ddm_stokes_base(par);
rng(27);
v=randn(size(a.U)); v=v/norm(v);
[J,dJ]=flow_drag_balance(par,S,a.U);
t=1e-6;
fp=flow_drag_balance(par,S,a.U+t*v);
fm=flow_drag_balance(par,S,a.U-t*v);
objective_gradient_error=abs((fp-fm)/(2*t)-dJ.'*v)/max(abs(dJ.'*v),1e-20);
assert(objective_gradient_error<1e-7,'Boundary functional gradient failed');
band=phi>.05 & phi<.95;
d=conv2(randn(size(phi)).*band,ones(3)/9,'same').*band;
d=d/max(abs(d(:))); directional=sum(a.sens(:).*d(:));
steps=[1e-3;5e-4;2.5e-4;1.25e-4]; remainder=zeros(size(steps));
for k=1:numel(steps)
    ap=p.fom(phi+steps(k)*d,a.U,false);
    remainder(k)=abs(ap.J-J-steps(k)*directional);
end
orders=log2(remainder(1:end-1)./remainder(2:end));
t=1e-6; ap=p.fom(phi+t*d,a.U,false); am=p.fom(phi-t*d,a.U,false);
fd=(ap.J-am.J)/(2*t);
adjoint_gradient_error=abs(fd-directional)/max(abs(fd),1e-20);
assert(adjoint_gradient_error<1e-5 && min(orders)>1.8, ...
    'Complete-drag design derivative failed');
o=struct('w_max',8,'r_max',8,'pod_tol',1e-12,'lsq','qr','gn_maxit',5);
rom=p.rom_new(o); rom=p.enrich(rom,a.U,a.W); b=p.rom_step(phi,rom,a.U);
rom_state_error=norm(b.U-a.U)/norm(a.U);
rom_adjoint_error=norm(b.W-a.W)/norm(a.W);
assert(max(rom_state_error,rom_adjoint_error)<1e-8,'ROM consistency failed');
report=table(objective_gradient_error,adjoint_gradient_error, ...
    min(orders),rom_state_error,rom_adjoint_error, ...
    'VariableNames',{'objective_gradient_error','adjoint_gradient_error', ...
    'minimum_Taylor_order','rom_state_error','rom_adjoint_error'});
out=fullfile(root,'results','flow_submission_traction');
if ~isfolder(out),mkdir(out);end
writetable(report,fullfile(out,'verification.csv'));
writetable(table(steps,remainder),fullfile(out,'Taylor_test.csv'));
disp(report);
end
