* PSP models - simple inverter
* Source: dwarning/VA-Models (BSD-3-Clause)
* https://github.com/dwarning/VA-Models/blob/main/examples/psp103/ngspice/psp_inverter.sp

.param Vcc = 1.2
.csparam vcc='Vcc'

* Model definitions (simplified for testing)
.model nch nmos level=69 version=103
+tnom=27 toxe=1.8e-9 vth0=0.4

.model pch pmos level=69 version=103
+tnom=27 toxe=1.8e-9 vth0=-0.4

* the voltage sources:
Vdd vdd gnd DC 'Vcc'
V1 in gnd pulse(0 'Vcc' 0p 200p 100p 1n 2n)
Vmeas vss 0 0

Xnot1 in vdd vss out not1

.subckt not1 a vdd vss z
nmp1  z a     vdd     vdd pch
+l=0.1u
+w=1u
+sa=0.0e+00
+sb=0.0e+00
+absource=1.0e-12
+lssource=1.0e-06
+lgsource=1.0e-06
+abdrain=1.0e-12
+lsdrain=1.0e-06
+lgdrain=1.0e-06
+mult=1.0e+00

nmn1  z a     vss     vss nch
+l=0.1u
+w=1u
+sa=0.0e+00
+sb=0.0e+00
+absource=1.0e-12
+lssource=1.0e-06
+lgsource=1.0e-06
+abdrain=1.0e-12
+lsdrain=1.0e-06
+lgdrain=1.0e-06
+mult=1.0e+00

c3  a     vss   0.384f
c2  z     vss   0.576f
.ends not1

* simulation command:
.tran 10ps 10ns
.dc V1 0 'vcc' 'vcc/100'

.control
run
plot in out
plot dc1.out
.endc

.end
