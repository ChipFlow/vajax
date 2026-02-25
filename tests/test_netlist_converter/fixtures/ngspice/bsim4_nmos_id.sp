* Sample netlist for BSIM4
* Source: dwarning/VA-Models (BSD-3-Clause)
* https://github.com/dwarning/VA-Models/blob/main/examples/bsim4/ngspice/nmos_id_vd_vg.sp

.include "modelcard_bsim4_nmos.mod"

* --- Voltage Sources ---
vd d  0 dc 50m
vg g  0 dc 1.0
vs s  0 dc 0.0
vb b  0 dc 0.0

* --- Transistor ---
NM1 d g s b n1 W = 10e-6 L = 1e-6 nf=1

* --- DC Analysis ---
.control
op
show all
dc vg 0.0 1.2 0.01 vb -0.5 0 0.1
plot -i(vd)
dc vd 0.0 1.2 0.01 vg 0.4 1.0 0.1
plot -i(vd)
.endc

.end
