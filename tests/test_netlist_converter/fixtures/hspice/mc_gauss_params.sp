* HSPICE Monte Carlo test with GAUSS/AGAUSS functions
* Tests preservation of GAUSS expressions in parameters

.PARAM LithOffset=AGAUSS(0u,0.1u,3)
.PARAM res_val=GAUSS(1k, 0.05, 3)
.PARAM vth_nom=0.4
.PARAM vth_sigma='0.01*vth_nom'
.PARAM vth_var='vth_nom+AGAUSS(0,vth_sigma,3)'

.subckt resistor_mc a b
R1 a b res_val
.ends resistor_mc

.subckt mos_mc d g s b
* MOSFET with lithography offset variation
M1 d g s b nmos w=1u l='100n+LithOffset'
.ends mos_mc

.end
