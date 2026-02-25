* Simple inverter for comparison testing
* No control block to avoid old converter issues

.param Vcc = 1.2

* Model definitions (BSIM4 level 54)
.model nch nmos level=54
+tnom=27 toxe=1.8e-9 vth0=0.4

.model pch pmos level=54
+tnom=27 toxe=1.8e-9 vth0=-0.4

* Subcircuit
.subckt inverter in out vdd vss
mp1 out in vdd vdd pch l=0.1u w=1u
mn1 out in vss vss nch l=0.1u w=1u
.ends inverter

.end
