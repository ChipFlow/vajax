* Ngspice Monte Carlo test with gauss/agauss functions
* Tests preservation of gauss expressions in parameters

.param vth_var='agauss(0, 0.05, 3)'
.param res_var='gauss(1k, 0.1, 3)'
.param mixed_expr='1.0 + agauss(0, 0.01, 3)'

.subckt inverter in out vdd vss
* MOS transistor with variation on threshold voltage
M1 out in vdd vdd pmos w=1u l=100n delvto='agauss(0, 0.1, 3)'
M2 out in vss vss nmos w=500n l=100n delvto='agauss(0, 0.1, 3)'
* Resistor with value variation
R1 vdd out 'gauss(1k, 0.05, 3)'
.ends inverter

.end
