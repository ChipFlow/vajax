"""SPICE raw file parser.

This module reads SPICE binary raw files (.raw) produced by simulators
like VACASK, ngspice, etc. It supports both real and complex data formats.

Origin: vendor/VACASK/python/rawfile.py
Author: VACASK project (https://github.com/arpadbuermen/VACASK)
License: Same as VACASK

Usage:
    from vajax.utils import rawread

    raw = rawread('simulation.raw')
    rf = raw.get()  # Get first plot

    t = rf['time']   # Get time vector by name
    v1 = rf['1']     # Get voltage at node "1"

    # For multi-sweep data:
    rf = raw.get(sweeps=1)
    for i in range(rf.sweepGroups):
        print(rf.sweepData(i))
"""

from typing import Dict, List, Tuple, Union

import numpy as np

__all__ = ["rawread", "RawFile", "RawData"]

BSIZE_SP = 512  # Max size of a line of data
HEADER_ENTRY_NAMES = {
    b"title",
    b"date",
    b"plotname",
    b"flags",
    b"no. variables",
    b"no. points",
    b"dimensions",
    b"command",
    b"option",
}


class RawFile:
    """Parsed SPICE raw file data."""

    def __init__(self, header: Dict, data: np.ndarray, sweeps: int = 0):
        self.title = header["title"].decode("ascii")
        self.date = header["date"].decode("ascii")
        self.plotname = header["plotname"].decode("ascii")
        self.flags = header["flags"].decode("ascii")
        self.names: List[str] = header["varnames"]
        self.units: List[str] = header["varunits"]
        self.data = data
        ndata, nvars = data.shape

        # Build name index
        self.nameToIndex: Dict[str, int] = {self.names[ii]: ii for ii in range(nvars)}

        # Split sweeps
        self.sweeps = sweeps
        if sweeps > 0:
            allends = np.array([], dtype="int64")
            for ii in range(sweeps):
                # ii-th outermost sweep
                var = data[:, ii]
                ends = np.where(var[1:] != var[:-1])[0] + 1
                ends = np.append(ends, ndata)
                allends = np.append(allends, ends)

            # Drop duplicates from array
            self.allends = np.unique(allends)

            # Sort
            self.allends.sort()

            # Beginnings
            self.allbegins = np.hstack([0, allends[:-1]])

            # Sweep groups
            self.sweepGroups = self.allbegins.size
        else:
            self.allbegins = np.array([0], dtype="int64")
            self.allends = np.array([ndata], dtype="int64")
            self.sweepGroups = 1

    def __getitem__(self, key: Union[str, int, Tuple[int, Union[str, int]]]) -> np.ndarray:
        """Get vector data by name or index.

        Args:
            key: Either a vector name/index, or (sweepGroup, vector) tuple

        Returns:
            Vector data as numpy array
        """
        if isinstance(key, tuple):
            # (sweepGroup, vector)
            sweepGroup, vec = key
            ii1 = self.allbegins[sweepGroup]
            ii2 = self.allends[sweepGroup]
        else:
            # vector - return all points of all sweeps
            ii1 = 0
            ii2 = self.data.shape[0]
            vec = key

        # Resolve vector name to index if needed
        if isinstance(vec, str):
            vec = self.nameToIndex[vec]

        return self.data[ii1:ii2, vec]

    def sweepData(self, sweepGroup: int) -> Dict[str, float]:
        """Get sweep parameter values for a sweep group."""
        ii = self.allbegins[sweepGroup]
        data = {}
        for jj in range(self.sweeps):
            data[self.names[jj]] = self.data[ii, jj]
        return data

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all vectors as a dictionary."""
        return {name: self[name] for name in self.names}


class RawData:
    """Container for multiple plots in a raw file."""

    def __init__(self, arrs: List[Tuple[Dict, np.ndarray]]):
        self.arrs = arrs

    def get(self, ndx: int = 0, sweeps: int = 0) -> RawFile:
        """Get a specific plot from the raw file.

        Args:
            ndx: Plot index (default 0 for first plot)
            sweeps: Number of sweep dimensions (0 for simple data)

        Returns:
            RawFile object with parsed data
        """
        plot, arr = self.arrs[ndx]
        return RawFile(plot, arr, sweeps)

    @property
    def num_plots(self) -> int:
        """Number of plots in the raw file."""
        return len(self.arrs)


def rawread(fname: str) -> RawData:
    """Read a SPICE binary raw file.

    Args:
        fname: Path to the .raw file

    Returns:
        RawData object containing parsed plots

    Raises:
        RuntimeError: If file cannot be read
        NotImplementedError: If file format is not supported (unpadded, ASCII)

    Example:
        >>> raw = rawread('tran.raw')
        >>> rf = raw.get()
        >>> t = rf['time']
        >>> v_out = rf['v(out)']
    """
    # Example header of raw file:
    # Title: rc band pass example circuit
    # Date: Sun Feb 21 11:29:14  2016
    # Plotname: AC Analysis
    # Flags: complex
    # No. Variables: 3
    # No. Points: 41
    # Variables:
    #         0       frequency       frequency       grid=3
    #         1       v(out)  voltage
    #         2       v(in)   voltage
    # Binary:

    with open(fname, "rb") as fp:
        plot: Dict = {}
        arrs: List[Tuple[Dict, np.ndarray]] = []

        while True:
            try:
                line = fp.readline(BSIZE_SP)
                if not line:
                    break
                splitLine = line.split(b":", maxsplit=1)
            except Exception as e:
                raise RuntimeError(f"Failed to read a line from file: {e}")

            if len(splitLine) == 2:
                key = splitLine[0].lower()

                # Ordinary header entries
                if key in HEADER_ENTRY_NAMES:
                    plot[key.decode("ascii")] = splitLine[1].strip()

                # Variable list
                if key == b"variables":
                    nvars = int(plot["no. variables"])
                    npoints = int(plot["no. points"])
                    plot["no. variables"] = nvars
                    plot["no. points"] = npoints
                    plot["varnames"] = []
                    plot["varunits"] = []

                    for ii in range(nvars):
                        # Get variable description, split it at spaces
                        txt = fp.readline(BSIZE_SP).strip().decode("ascii")
                        varDesc = txt.split(maxsplit=3)
                        if len(varDesc) > 3 and "dims" in varDesc[3]:
                            raise NotImplementedError(
                                "Raw files with different length vectors are not supported."
                            )
                        # Check variable numbering
                        assert ii == int(varDesc[0])
                        # Get name and units
                        plot["varnames"].append(varDesc[1])
                        plot["varunits"].append(varDesc[2])

                # Binary data start
                if key == b"binary":
                    # Check for unpadded
                    if b"unpadded" in plot["flags"]:
                        raise NotImplementedError("Unpadded raw files are not supported.")

                    dtype = np.complex128 if b"complex" in plot["flags"] else np.float64
                    arr = np.fromfile(fp, dtype=dtype, count=npoints * nvars)
                    arr = arr.reshape((npoints, nvars))
                    arrs.append((plot.copy(), arr))
                    fp.readline()  # Read to the end of line
                    plot = {}  # Reset for next plot

                if key == b"ascii":
                    raise NotImplementedError("ASCII raw files are not supported.")
            else:
                # Header line does not have two parts, check for end
                if not line.strip():
                    continue  # Skip empty lines

    return RawData(arrs)
