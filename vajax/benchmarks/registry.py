"""VACASK Benchmark Registry.

Auto-discovers benchmarks from vendor/VACASK/benchmark/*/vacask/runme.sim
and parses analysis parameters from .sim files.

Usage:
    from vajax.benchmarks.registry import discover_benchmarks, BENCHMARKS

    # Auto-discover all benchmarks
    benchmarks = discover_benchmarks()

    # Or use pre-discovered registry
    for name, info in BENCHMARKS.items():
        print(f"{name}: dt={info.dt:.2e}, t_stop={info.t_stop:.2e}")
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Project root and benchmark directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "vendor" / "VACASK" / "benchmark"
ADDITIONAL_BENCHMARKS_DIR = PROJECT_ROOT / "vajax" / "benchmarks" / "data"


@dataclass
class BenchmarkInfo:
    """Information about a VACASK benchmark circuit."""

    name: str
    sim_path: Path
    dt: float  # Time step (step= parameter)
    t_stop: float  # Stop time (stop= parameter)
    devices: Set[str] = field(default_factory=set)  # Device types used
    title: str = ""  # First line of .sim file
    icmode: Optional[str] = None  # Initial condition mode (uic, etc.)
    max_steps: Optional[int] = None  # Optional max steps override
    skip: bool = False
    skip_reason: str = ""
    xfail: bool = False
    xfail_reason: str = ""
    gpu_min_vram_gb: int = 0  # Minimum GPU VRAM in GB (0 = no requirement)

    @property
    def uses_mosfet(self) -> bool:
        """Check if benchmark uses MOSFET devices."""
        return "psp103" in self.devices or "mosfet" in self.devices

    @property
    def uses_diode(self) -> bool:
        """Check if benchmark uses diode devices."""
        return "diode" in self.devices

    @property
    def is_large(self) -> bool:
        """Check if benchmark is large (should use sparse solver)."""
        # c6288 has ~25k nodes, mul64 has ~400k+ nodes
        return self.name in ("c6288", "mul64")


def parse_si_value(s: str) -> float:
    """Parse SPICE value with SI suffix.

    Args:
        s: String like "1m", "100n", "10k", "0.05n", etc.

    Returns:
        Float value with SI scaling applied
    """
    s = s.strip().lower()

    # Order matters - check longer suffixes first
    suffixes = [
        ("meg", 1e6),
        ("mil", 25.4e-6),
        ("ms", 1e-3),
        ("us", 1e-6),
        ("ns", 1e-9),
        ("ps", 1e-12),
        ("fs", 1e-15),
        ("f", 1e-15),
        ("p", 1e-12),
        ("n", 1e-9),
        ("u", 1e-6),
        ("m", 1e-3),
        ("k", 1e3),
        ("g", 1e9),
        ("t", 1e12),
    ]

    for suffix, mult in suffixes:
        if s.endswith(suffix):
            return float(s[: -len(suffix)]) * mult
    return float(s)


def parse_analysis_params(sim_path: Path) -> Tuple[float, float, Optional[str]]:
    """Extract step, stop, and icmode from analysis line in .sim file.

    Handles formats like:
        analysis tran1 tran step=0.05n stop=1u maxstep=0.05n
        analysis tranmul tran stop=2n step=2p icmode="uic"

    Args:
        sim_path: Path to .sim file

    Returns:
        (dt, t_stop, icmode) tuple
    """
    content = sim_path.read_text()

    # Find analysis line(s) - take the first tran analysis
    # Pattern: analysis <name> tran <params>
    analysis_pattern = r"analysis\s+\w+\s+tran\s+(.+?)(?:\n|$)"
    match = re.search(analysis_pattern, content, re.IGNORECASE)

    if not match:
        raise ValueError(f"No transient analysis found in {sim_path}")

    params_str = match.group(1)

    # Extract step=, stop=, icmode=
    step_match = re.search(r"step\s*=\s*([^\s]+)", params_str, re.IGNORECASE)
    stop_match = re.search(r"stop\s*=\s*([^\s]+)", params_str, re.IGNORECASE)
    icmode_match = re.search(r'icmode\s*=\s*"?(\w+)"?', params_str, re.IGNORECASE)

    if not step_match:
        raise ValueError(f"No step= parameter found in {sim_path}")
    if not stop_match:
        raise ValueError(f"No stop= parameter found in {sim_path}")

    dt = parse_si_value(step_match.group(1))
    t_stop = parse_si_value(stop_match.group(1))
    icmode = icmode_match.group(1) if icmode_match else None

    return dt, t_stop, icmode


def detect_devices(sim_path: Path) -> Set[str]:
    """Detect device types from load statements in .sim file.

    Args:
        sim_path: Path to .sim file

    Returns:
        Set of device type strings (e.g., {"resistor", "capacitor", "psp103"})
    """
    content = sim_path.read_text()
    devices = set()

    # Find all load statements
    # Pattern: load "path/to/device.osdi"
    load_pattern = r'load\s+"([^"]+)"'

    for match in re.finditer(load_pattern, content, re.IGNORECASE):
        osdi_path = match.group(1).lower()

        # Map OSDI file names to device types
        if "psp103" in osdi_path:
            devices.add("psp103")
        elif "resistor" in osdi_path:
            devices.add("resistor")
        elif "capacitor" in osdi_path:
            devices.add("capacitor")
        elif "diode" in osdi_path:
            devices.add("diode")
        elif "inductor" in osdi_path:
            devices.add("inductor")

    # Also check for vsource/isource models
    if re.search(r"model\s+\w+\s+vsource", content, re.IGNORECASE):
        devices.add("vsource")
    if re.search(r"model\s+\w+\s+isource", content, re.IGNORECASE):
        devices.add("isource")

    return devices


def get_title(sim_path: Path) -> str:
    """Get benchmark title from first line of .sim file."""
    with open(sim_path) as f:
        first_line = f.readline().strip()
        # Remove comment markers if present
        if first_line.startswith("//"):
            first_line = first_line[2:].strip()
        return first_line


def _parse_benchmark(sim_path: Path, name: str) -> BenchmarkInfo:
    """Parse a single benchmark .sim file into BenchmarkInfo."""
    try:
        dt, t_stop, icmode = parse_analysis_params(sim_path)
        devices = detect_devices(sim_path)
        title = get_title(sim_path)

        info = BenchmarkInfo(
            name=name,
            sim_path=sim_path,
            dt=dt,
            t_stop=t_stop,
            devices=devices,
            title=title,
            icmode=icmode,
        )

        # Apply known constraints
        if name == "c6288":
            # Large circuit needs sparse solver and limited steps for CI
            info.max_steps = 20
            info.xfail = True
            info.xfail_reason = "Node count mismatch - need node collapse"
        elif name == "mul64":
            # Very large circuit (~266k MOSFETs, ~666k unknowns)
            # Requires >16GB GPU VRAM for cuDSS sparse factorization
            info.max_steps = 5
            info.gpu_min_vram_gb = 24
        elif name == "tb_dp":
            # Large SRAM, should use sparse solver
            info.max_steps = 10
        elif name == "mul":
            # Diode cascade circuit is numerically stiff, needs adaptive timestep
            pass  # Use adaptive=True in tests for this benchmark

        return info

    except Exception as e:
        # Return a skipped benchmark on parse error
        return BenchmarkInfo(
            name=name,
            sim_path=sim_path,
            dt=0,
            t_stop=0,
            skip=True,
            skip_reason=f"Parse error: {e}",
        )


def discover_benchmarks(benchmark_dir: Optional[Path] = None) -> Dict[str, BenchmarkInfo]:
    """Auto-discover benchmarks from multiple locations.

    Discovers from:
    1. vendor/VACASK/benchmark/*/vacask/runme.sim
    2. vajax/benchmarks/data/*/*.sim

    Args:
        benchmark_dir: Optional override for VACASK benchmark directory

    Returns:
        Dict mapping benchmark name to BenchmarkInfo
    """
    if benchmark_dir is None:
        benchmark_dir = BENCHMARK_DIR

    benchmarks = {}

    # Discover VACASK benchmarks
    if benchmark_dir.exists():
        for sim_path in benchmark_dir.glob("*/vacask/runme.sim"):
            name = sim_path.parent.parent.name  # e.g., "ring" from .../ring/vacask/runme.sim
            benchmarks[name] = _parse_benchmark(sim_path, name)

    # Discover additional benchmarks from vajax/benchmarks/data/
    if ADDITIONAL_BENCHMARKS_DIR.exists():
        for sim_path in ADDITIONAL_BENCHMARKS_DIR.glob("*/*.sim"):
            # Use parent directory name as benchmark name
            name = sim_path.parent.name  # e.g., "tb_dp" from .../tb_dp/file.sim
            if name not in benchmarks:  # Don't override VACASK benchmarks
                benchmarks[name] = _parse_benchmark(sim_path, name)

    return benchmarks


def get_benchmark(name: str) -> Optional[BenchmarkInfo]:
    """Get a specific benchmark by name.

    Args:
        name: Benchmark name (e.g., "ring", "rc", "c6288")

    Returns:
        BenchmarkInfo or None if not found
    """
    return BENCHMARKS.get(name)


def list_benchmarks() -> list[str]:
    """List all available benchmark names."""
    return list(BENCHMARKS.keys())


def get_runnable_benchmarks() -> Dict[str, BenchmarkInfo]:
    """Get benchmarks that can be run (not skipped)."""
    return {name: info for name, info in BENCHMARKS.items() if not info.skip}


# Pre-discover benchmarks on module import
BENCHMARKS = discover_benchmarks()
