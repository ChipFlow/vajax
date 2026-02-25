* Simple subcircuit test for comparison testing
* Uses only instance types supported by both converters

.param Vcc = 1.2

* Subcircuit definition
.subckt buffer in out vdd vss
R1 in mid 1k
R2 mid out 2k
C1 mid vss 10p
.ends buffer

* Top-level instance
Xbuf1 input output vdd vss buffer

.end
