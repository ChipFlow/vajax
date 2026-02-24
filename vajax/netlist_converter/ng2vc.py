#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SPICE to VACASK netlist converter.

Supports multiple SPICE dialects: Ngspice (default), HSPICE, LTSpice.

Usage:
    python3 -m ng2vc [<args>] <input file> [<output file>]
    python3 -m ng2vc --dialect hspice input.hspice output.vc

If no output file is provided, the converted netlist is printed to stdout.
"""

import sys

from .ng2vclib import dfl
from .ng2vclib.converter import Converter
from .spiceparser.dialect import detect_dialect_from_file

HELP_TEXT = """\
SPICE to VACASK netlist converter.

Usage: python3 -m ng2vc [<args>] <input file> [<output file>]

If no output file is provided, the converted netlist is printed to
the standard output.

If output file is a relative path it is considered relative to
the input file.

Creates destination directory, if needed.

Arguments:
  -h --help           print help
  -d  --dialect       SPICE dialect: auto, ngspice (default), hspice, ltspice
                      auto detects dialect from file content
  -dr --read-depth    maximal depth to which sources are loaded
                      (infinite by default)
  -dp --process-depth maximal depth to which sources are processed
                      (infinite by default)
  -do --output-depth  maximal depth to which sources are output
                      (infinite by default)
  -sp --sourcepath    add a directory to the sourcepath where
                      .include and .lib files are found. By default
                      sourcepath already contains the current directory.
  -am --all-models    dumps all models, not just those that are used
                      (disabled by default)
"""


def main():
    """Main entry point for ng2vc converter."""
    ndx = 1
    from_file = None
    to_file = None
    dialect = "ngspice"
    read_depth = None
    process_depth = None
    output_depth = None
    sourcepath = ["."]
    all_models = False

    while ndx < len(sys.argv):
        arg = sys.argv[ndx]
        if arg.startswith("-"):
            if arg in ("--help", "-h"):
                print(HELP_TEXT)
                return 0
            elif arg in ("-d", "--dialect"):
                if ndx + 1 >= len(sys.argv):
                    print("Too few arguments.", file=sys.stderr)
                    return 1
                ndx += 1
                dialect = sys.argv[ndx].lower()
                if dialect not in ("auto", "ngspice", "hspice", "ltspice"):
                    print(f"Unknown dialect: {dialect}", file=sys.stderr)
                    print("Valid dialects: auto, ngspice, hspice, ltspice", file=sys.stderr)
                    return 1
            elif arg in ("-dr", "--read-depth"):
                if ndx + 1 >= len(sys.argv):
                    print("Too few arguments.", file=sys.stderr)
                    return 1
                ndx += 1
                read_depth = int(sys.argv[ndx])
            elif arg in ("-dp", "--process-depth"):
                if ndx + 1 >= len(sys.argv):
                    print("Too few arguments.", file=sys.stderr)
                    return 1
                ndx += 1
                process_depth = int(sys.argv[ndx])
            elif arg in ("-do", "--output-depth"):
                if ndx + 1 >= len(sys.argv):
                    print("Too few arguments.", file=sys.stderr)
                    return 1
                ndx += 1
                output_depth = int(sys.argv[ndx])
            elif arg in ("-sp", "--sourcepath"):
                if ndx + 1 >= len(sys.argv):
                    print("Too few arguments.", file=sys.stderr)
                    return 1
                ndx += 1
                sourcepath.append(sys.argv[ndx])
            elif arg in ("-am", "--all-models"):
                all_models = True
            else:
                print(f"Unknown argument: {arg}", file=sys.stderr)
                print(HELP_TEXT, file=sys.stderr)
                return 1
        else:
            # Positional argument: input file
            if ndx + 1 > len(sys.argv):
                print("Need input file.", file=sys.stderr)
                print(HELP_TEXT, file=sys.stderr)
                return 1

            from_file = arg

            if ndx + 2 < len(sys.argv):
                print("Too many arguments.", file=sys.stderr)
                print(HELP_TEXT, file=sys.stderr)
                return 1

            if ndx + 2 == len(sys.argv):
                to_file = sys.argv[ndx + 1]
            else:
                to_file = None
            break

        ndx += 1

    if from_file is None:
        print("Need input file.", file=sys.stderr)
        print(HELP_TEXT, file=sys.stderr)
        return 1

    # Build configuration
    cfg = dfl.default_config()
    cfg["sourcepath"] = sourcepath
    cfg["read_depth"] = read_depth
    cfg["process_depth"] = process_depth
    cfg["output_depth"] = output_depth
    cfg["all_models"] = all_models

    # Auto-detect dialect if requested
    if dialect == "auto":
        dialect = detect_dialect_from_file(from_file)

    # Create converter with specified dialect
    converter = Converter(cfg, dialect=dialect)
    converter.convert(from_file, to_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
