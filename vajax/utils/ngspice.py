"""ngspice utilities for running simulations and parsing output.

Provides functions to:
- Find the ngspice binary
- Run ngspice simulations
- Parse ngspice .control section parameters
- Generate raw output files
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "find_ngspice_binary",
    "parse_control_section",
    "run_ngspice",
    "NgspiceError",
    "validate_path_safe",
]


def validate_path_safe(path: Path, allowed_parent: Optional[Path] = None) -> bool:
    """Validate that a path is safe (no directory traversal attacks).

    Checks that:
    1. Path resolves to an absolute path without '..' components
    2. If allowed_parent is specified, path must be within that directory

    Args:
        path: Path to validate
        allowed_parent: Optional parent directory that path must be within

    Returns:
        True if path is safe, False otherwise

    Example:
        >>> validate_path_safe(Path("circuit.sim"))
        True
        >>> validate_path_safe(Path("../../../etc/passwd"))
        False
        >>> validate_path_safe(Path("sub/circuit.sim"), Path("/project"))
        True  # if sub/circuit.sim is within /project
    """
    try:
        resolved = path.resolve()

        # Check for suspicious path components in original path string
        path_str = str(path)
        if ".." in path_str:
            logger.warning(f"Path contains '..': {path}")
            return False

        if allowed_parent is not None:
            parent_resolved = allowed_parent.resolve()
            try:
                resolved.relative_to(parent_resolved)
            except ValueError:
                logger.warning(f"Path {resolved} is not within allowed parent {parent_resolved}")
                return False

        return True

    except (OSError, ValueError) as e:
        logger.warning(f"Path validation failed for {path}: {e}")
        return False


class NgspiceError(Exception):
    """Error running ngspice simulation."""

    pass


def find_ngspice_binary() -> Optional[Path]:
    """Find the ngspice binary in standard locations.

    Checks in order:
    1. NGSPICE_BIN environment variable
    2. System PATH via shutil.which
    3. Common installation paths

    Returns:
        Path to ngspice binary or None if not found
    """
    # Check environment variable
    env_path = os.environ.get("NGSPICE_BIN")
    if env_path:
        path = Path(env_path).resolve()
        if path.exists() and path.is_file():
            return path

    # Check PATH
    which_result = shutil.which("ngspice")
    if which_result:
        return Path(which_result)

    # Check common locations
    search_paths = [
        Path("/usr/local/bin/ngspice"),
        Path("/usr/bin/ngspice"),
        Path("/opt/homebrew/bin/ngspice"),  # macOS Homebrew ARM
        Path("/opt/local/bin/ngspice"),  # MacPorts
    ]
    for path in search_paths:
        if path.exists():
            return path

    return None


def parse_control_section(netlist_path: Path) -> Dict[str, Any]:
    """Parse .control section from ngspice netlist.

    Extracts analysis parameters from the .control section including:
    - tran: step, stop, start, maxstep
    - ac: type, points, fstart, fstop
    - dc: srcnam, vstart, vstop, vincr
    - osdi: pre-loaded OSDI model file

    Args:
        netlist_path: Path to ngspice netlist file

    Returns:
        Dict with parsed control section parameters:
        - 'analysis': analysis type ('tran', 'ac', 'dc', 'op')
        - 'step', 'stop', 'start', 'maxstep': for tran
        - 'osdi': OSDI model path if present
    """
    content = netlist_path.read_text()
    result: Dict[str, Any] = {}

    # Find .control section
    control_match = re.search(r"\.control\s+(.*?)\.endc", content, re.DOTALL | re.IGNORECASE)
    if not control_match:
        # No .control section, check for dot commands
        return _parse_dot_commands(content)

    control = control_match.group(1)

    # Parse tran command: tran step stop [start [maxstep]]
    tran_match = re.search(
        r"\btran\s+(\S+)\s+(\S+)(?:\s+(\S+))?(?:\s+(\S+))?", control, re.IGNORECASE
    )
    if tran_match:
        result["analysis"] = "tran"
        result["step"] = tran_match.group(1)
        result["stop"] = tran_match.group(2)
        result["start"] = tran_match.group(3) or "0"
        result["maxstep"] = tran_match.group(4)

    # Parse ac command: ac type points fstart fstop
    ac_match = re.search(r"\bac\s+(dec|oct|lin)\s+(\S+)\s+(\S+)\s+(\S+)", control, re.IGNORECASE)
    if ac_match:
        result["analysis"] = "ac"
        result["type"] = ac_match.group(1)
        result["points"] = ac_match.group(2)
        result["fstart"] = ac_match.group(3)
        result["fstop"] = ac_match.group(4)

    # Parse dc command: dc srcnam vstart vstop vincr
    dc_match = re.search(r"\bdc\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", control, re.IGNORECASE)
    if dc_match:
        result["analysis"] = "dc"
        result["srcnam"] = dc_match.group(1)
        result["vstart"] = dc_match.group(2)
        result["vstop"] = dc_match.group(3)
        result["vincr"] = dc_match.group(4)

    # Parse op command
    if re.search(r"\bop\b", control, re.IGNORECASE):
        if "analysis" not in result:
            result["analysis"] = "op"

    # Parse OSDI pre-load: pre_osdi path/to/model.osdi
    osdi_match = re.search(r"pre_osdi\s+(\S+)", control, re.IGNORECASE)
    if osdi_match:
        result["osdi"] = osdi_match.group(1)

    return result


def _parse_dot_commands(content: str) -> Dict[str, Any]:
    """Parse analysis from .TRAN/.AC/.DC dot commands outside .control."""
    result: Dict[str, Any] = {}

    # Process line by line to avoid cross-line matching issues
    for line in content.split("\n"):
        line = line.strip()

        # .TRAN step stop [start [maxstep]]
        tran_match = re.match(
            r"\.TRAN\s+(\S+)\s+(\S+)(?:\s+(\S+))?(?:\s+(\S+))?", line, re.IGNORECASE
        )
        if tran_match:
            result["analysis"] = "tran"
            result["step"] = tran_match.group(1)
            result["stop"] = tran_match.group(2)
            start = tran_match.group(3)
            maxstep = tran_match.group(4)
            # Only use start/maxstep if they're numeric values
            if start and _is_numeric_value(start):
                result["start"] = start
            else:
                result["start"] = "0"
            if maxstep and _is_numeric_value(maxstep):
                result["maxstep"] = maxstep
            continue

        # .AC type points fstart fstop
        ac_match = re.match(r"\.AC\s+(dec|oct|lin)\s+(\S+)\s+(\S+)\s+(\S+)", line, re.IGNORECASE)
        if ac_match:
            result["analysis"] = "ac"
            result["type"] = ac_match.group(1)
            result["points"] = ac_match.group(2)
            result["fstart"] = ac_match.group(3)
            result["fstop"] = ac_match.group(4)
            continue

    return result


def _is_numeric_value(s: str) -> bool:
    """Check if a string looks like a numeric SPICE value.

    Args:
        s: String to check

    Returns:
        True if it looks like a number (with optional SI suffix)
    """
    if not s:
        return False
    # Must start with a digit or minus sign
    if not (s[0].isdigit() or s[0] == "-" or s[0] == "."):
        return False
    # Common SI suffixes
    suffixes = {"t", "g", "meg", "k", "m", "u", "n", "p", "f", "a"}
    s_lower = s.lower()
    # Strip any suffix and check if remainder is numeric
    for suffix in sorted(suffixes, key=len, reverse=True):
        if s_lower.endswith(suffix):
            s_lower = s_lower[: -len(suffix)]
            break
    # Also handle 'ms', 'us', 'ns', 'ps', 'fs'
    for suffix in ["ms", "us", "ns", "ps", "fs"]:
        if s_lower.endswith(suffix):
            s_lower = s_lower[: -len(suffix)]
            break
    try:
        float(s_lower)
        return True
    except ValueError:
        return False


def run_ngspice(
    netlist_path: Path,
    output_dir: Optional[Path] = None,
    ngspice_bin: Optional[Path] = None,
    timeout: int = 120,
) -> Tuple[Optional[Path], Optional[str]]:
    """Run ngspice on a netlist and return the raw file path.

    This function:
    1. Copies the netlist to a temp directory
    2. Modifies the .control section to write a raw file
    3. Runs ngspice in batch mode
    4. Returns the path to the generated raw file

    Args:
        netlist_path: Path to ngspice netlist
        output_dir: Directory for output files (default: temp dir)
        ngspice_bin: Path to ngspice binary (default: auto-detect)
        timeout: Timeout in seconds

    Returns:
        (raw_file_path, error) tuple:
        - raw_file_path: Path to generated raw file, or None on failure
        - error: Error message, or None on success
    """
    # Validate netlist path (no directory traversal)
    if not validate_path_safe(netlist_path):
        return None, f"Invalid netlist path: {netlist_path}"

    if ngspice_bin is None:
        ngspice_bin = find_ngspice_binary()
        if ngspice_bin is None:
            return None, "ngspice binary not found. Install ngspice or set NGSPICE_BIN."

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="ngspice_"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read original netlist
    try:
        content = netlist_path.read_text()
    except Exception as e:
        return None, f"Failed to read netlist: {e}"

    # Modify netlist to save raw file
    content = _inject_raw_output(content)

    # Write modified netlist
    modified_netlist = output_dir / netlist_path.name
    modified_netlist.write_text(content)

    # Copy any .include files to output directory
    _copy_includes(netlist_path, output_dir, content)

    try:
        result = subprocess.run(
            [str(ngspice_bin), "-b", str(modified_netlist)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check for raw file
        raw_path = output_dir / "output.raw"
        if raw_path.exists():
            return raw_path, None

        # Check for other .raw files
        raw_files = list(output_dir.glob("*.raw"))
        if raw_files:
            return raw_files[0], None

        # No raw file produced
        if result.returncode != 0:
            stderr = result.stderr[:1000] if result.stderr else ""
            stdout = result.stdout[:1000] if result.stdout else ""
            return None, f"ngspice failed (exit {result.returncode}):\n{stderr}\n{stdout}"

        return None, f"No raw file produced. stdout: {result.stdout[:500]}"

    except subprocess.TimeoutExpired:
        return None, f"ngspice timed out after {timeout}s"
    except Exception as e:
        return None, f"Error running ngspice: {e}"


def _inject_raw_output(content: str) -> str:
    """Inject raw output commands into netlist .control section.

    If a .control section exists, adds 'write output.raw all' before 'quit'.
    If no .control section, adds one with run and write commands.

    Args:
        content: Original netlist content

    Returns:
        Modified netlist content
    """
    # Check if .control section exists
    if re.search(r"\.control", content, re.IGNORECASE):
        # Find quit command and insert before it
        if re.search(r"\bquit\b", content, re.IGNORECASE):
            content = re.sub(
                r"(\s*)(quit)",
                r"\1set wr_vecnames\n\1write output.raw all\n\1\2",
                content,
                flags=re.IGNORECASE,
            )
        else:
            # No quit, add write before .endc
            content = re.sub(
                r"(\.endc)",
                r"set wr_vecnames\nwrite output.raw all\nquit\n\1",
                content,
                flags=re.IGNORECASE,
            )
    else:
        # No .control section - add one before .end
        end_match = re.search(r"\.end\b", content, re.IGNORECASE)
        if end_match:
            insert_pos = end_match.start()
            control_section = """
.control
set wr_vecnames
run
write output.raw all
quit
.endc
"""
            content = content[:insert_pos] + control_section + content[insert_pos:]

    return content


def _copy_includes(netlist_path: Path, output_dir: Path, content: str) -> None:
    """Copy .include files and OSDI model files to output directory.

    Only copies files that are within the source directory tree
    to prevent path traversal attacks.

    Args:
        netlist_path: Original netlist path
        output_dir: Destination directory
        content: Netlist content
    """
    source_dir = netlist_path.parent.resolve()

    def copy_referenced_file(file_path: str, file_type: str) -> None:
        """Copy a referenced file to output directory."""
        # Resolve relative to source directory
        if not Path(file_path).is_absolute():
            src_file = source_dir / file_path
        else:
            src_file = Path(file_path)

        # Validate path is safe (within source directory or absolute path exists)
        if not validate_path_safe(src_file, allowed_parent=source_dir):
            # For absolute paths, just check they don't have traversal
            if Path(file_path).is_absolute():
                if not validate_path_safe(src_file):
                    logger.warning(f"Skipping unsafe {file_type} path: {file_path}")
                    return
            else:
                logger.warning(f"Skipping {file_type} path outside source dir: {file_path}")
                return

        if src_file.exists():
            dst_file = output_dir / src_file.name
            if not dst_file.exists():
                try:
                    shutil.copy2(src_file, dst_file)
                    logger.debug(f"Copied {file_type} file: {src_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to copy {file_type} file {src_file}: {e}")

    # Find .include directives
    for match in re.finditer(r'\.include\s+["\']?([^"\'\s]+)["\']?', content, re.IGNORECASE):
        copy_referenced_file(match.group(1), "include")

    # Find pre_osdi directives (OSDI model loading)
    for match in re.finditer(r'pre_osdi\s+["\']?([^"\'\s]+)["\']?', content, re.IGNORECASE):
        copy_referenced_file(match.group(1), "OSDI")
