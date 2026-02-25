* HSPICE Inverter Testbench
* Demonstrates .lib usage and HSPICE-style parameters

.lib 'nmos_library.lib' typical

.param vdd_val = 1.8
.param wp = 1u
.param wn = 500n
.param lmin = 100n

* Supply
Vdd vdd 0 dc vdd_val
Vss vss 0 dc 0

* Input stimulus
Vin in 0 pulse(0 vdd_val 0 100p 100p 5n 10n)

* Inverter subcircuit
.subckt inv in out vdd vss params: wp=1u wn=500n l=100n
Mp out in vdd vdd pmos_typ w=wp l=l
Mn out in vss vss nmos_typ w=wn l=l
.ends inv

* Instance
Xinv1 in out vdd vss inv wp=wp wn=wn l=lmin

* Load capacitor
Cload out 0 10f

* Analysis
.tran 10p 50n
.dc Vin 0 vdd_val 0.01

.option post=2 accurate

.end
