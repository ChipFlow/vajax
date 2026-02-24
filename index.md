# VAJAX CI Summary

_Last updated: 2026-02-24 02:08 UTC_

_Commit: [ce571fd1](https://github.com/ChipFlow/vajax/commit/ce571fd10194fa1e408f621a1e218236c777cce2)_

## Test Coverage

| Suite | Passed | Failed | Errors | Skipped | Total | Time |
|-------|--------|--------|--------|---------|-------|------|
| benchmark-dense | 5 | 0 | 0 | 1 | 6 | 369.8s | PASS |
| benchmark-sparse | 6 | 0 | 0 | 1 | 7 | 382.2s | PASS |
| benchmarks | 5 | 0 | 0 | 0 | 5 | 366.9s | PASS |
| ngspice | 0 | 0 | 0 | 71 | 71 | 246.4s | PASS |
| openvaf-py | 320 | 0 | 0 | 92 | 412 | 1024.0s | PASS |
| unit | 72 | 0 | 0 | 5 | 77 | 21.1s | PASS |
| xyce | 36 | 0 | 0 | 1893 | 1929 | 268.0s | PASS |
| **Total** | **444** | **0** | **0** | **2063** | **2507** | 2678.4s |


## Performance

### CPU Benchmarks

| Benchmark | Steps | VAJAX (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| rc | 1,000,000 | 0.0135 | 0.0019 | 7.03x | 4.5s |
| graetz | 1,000,000 | 0.0194 | 0.0038 | 5.17x | 10.4s |
| mul | 500,000 | 0.0416 | 0.0038 | 11.03x | 7.5s |
| ring | 19,999 | 0.5642 | 0.1086 | 5.19x | 165.1s |
| tb_dp | 299 | 0.1048 | N/A | N/A | 5.5s |

_No gpu benchmarks benchmark data available._


---

[View workflows](https://github.com/ChipFlow/vajax/actions) | 
[Repository](https://github.com/ChipFlow/vajax)
