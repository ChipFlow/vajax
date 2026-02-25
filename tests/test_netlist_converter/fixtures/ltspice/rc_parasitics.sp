* LTSpice RC circuit with component parasitics
* Demonstrates LTSpice-specific passive component parameters

* Input voltage source
V1 in 0 DC 1 AC 1

* Resistor with series parasitic resistance
R1 in mid 1k Rser=10

* Capacitor with ESR, ESL, and parallel resistance
C1 mid out 1u Rser=100m Lser=1n Rpar=1meg

* Inductor with parasitics
L1 out gnd 10u Rser=50m Rpar=100k Cpar=1p

* Load resistor
Rload out gnd 10k

* Analysis commands
.ac dec 100 1 1meg
.tran 1u 10m

* LTSpice-specific options
.backanno
.end
