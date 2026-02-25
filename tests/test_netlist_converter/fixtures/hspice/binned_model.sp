* HSPICE binned model example
* Demonstrates .if/.elseif/.else/.endif for bin selection

.param l_drawn = 100n
.param w_drawn = 1u

* Bin selection based on geometry
.if (l_drawn < 200n)
.param vth0_val = 0.4
.elseif (l_drawn < 500n)
.param vth0_val = 0.38
.else
.param vth0_val = 0.35
.endif

* Simple passive circuit (no MOSFETs to avoid model lookup issues)
R1 in out 1k
C1 out 0 1u

.end
