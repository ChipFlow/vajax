# VAJAX CI Summary

_Last updated: 2026-02-24 01:44 UTC_

_Commit: [2cb5651b](https://github.com/ChipFlow/vajax/commit/2cb5651be1331fcb08506718e91e5fb270ee4490)_

## Test Coverage

| Suite | Passed | Failed | Errors | Skipped | Total | Time |
|-------|--------|--------|--------|---------|-------|------|
| benchmark-dense | 5 | 0 | 0 | 1 | 6 | 539.8s | PASS |
| benchmark-sparse | 6 | 0 | 0 | 1 | 7 | 562.9s | PASS |
| benchmarks | 5 | 0 | 0 | 0 | 5 | 748.4s | PASS |
| ngspice | 0 | 0 | 0 | 71 | 71 | 250.3s | PASS |
| openvaf-py | 320 | 0 | 0 | 92 | 412 | 1052.6s | PASS |
| unit | 72 | 0 | 0 | 5 | 77 | 20.9s | PASS |
| xyce | 36 | 0 | 0 | 1893 | 1929 | 258.1s | PASS |
| **Total** | **444** | **0** | **0** | **2063** | **2507** | 3433.0s |


## Performance

### CPU Benchmarks

| Benchmark | Steps | VAJAX (ms/step) | VACASK (ms/step) | Ratio | Startup |
|-----------|-------|---------------------|------------------|-------|---------|
| rc | 1,000,000 | 0.0125 | 0.0019 | 6.74x | 3.6s |
| graetz | 1,000,000 | 0.0197 | 0.0038 | 5.25x | 9.6s |
| mul | 500,000 | 0.0418 | 0.0039 | 10.82x | 7.1s |
| ring | 19,999 | 0.5460 | 0.1087 | 5.02x | 155.0s |
| tb_dp | 299 | 0.1022 | N/A | N/A | 5.0s |

_No gpu benchmarks benchmark data available._


---

[View workflows](https://github.com/ChipFlow/vajax/actions) | 
[Repository](https://github.com/ChipFlow/vajax)
