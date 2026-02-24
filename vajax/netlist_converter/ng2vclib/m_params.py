# SPDX-FileCopyrightText: 2025 ChipFlow
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Parameter handling mixin for ng2vc converter.

This module uses shared functions from ..spiceparser.params where possible,
providing a backward-compatible interface for the ng2vclib converter.
"""

# Import shared parameter utilities
from ..spiceparser.params import (
    convert_si_prefixes as si_replace,
)
from ..spiceparser.params import (
    format_value as _format_value,
)
from ..spiceparser.params import (
    merge_vector_params,
)
from ..spiceparser.params import (
    process_expressions as _process_expressions,
)
from ..spiceparser.params import (
    process_terminals as _process_terminals,
)
from ..spiceparser.params import (
    split_params as _split_params,
)
from .exc import ConverterError


class ParamsMixin:
    """Mixin providing parameter handling methods for the Converter.

    These methods delegate to shared functions from ..spiceparser.params
    while maintaining backward compatibility with existing ng2vclib code.
    """

    def format_extra_params(self, extras, lws, indent):
        """Format extra parameters.

        For now, we do not care about line length.
        """
        intxt = lws + (" " * indent)

        txt = ""
        first = True
        for name, value in extras.items():
            if not first:
                txt += " "
            first = False
            txt += name + "=" + value

        return intxt + txt

    def format_value(self, valuestr):
        """Format a value. Removes curly braces. Changes SI prefixes.

        Delegates to spiceparser.params.format_value().
        """
        return _format_value(valuestr)

    def format_params(self, params, atcol=0, indent=2):
        """Format parameters, indent them, make sure lines are not too long.

        *atcol* is the column number where we start to dump parameters.

        Does not indent. Uses *indent* for computing when to split a line.

        Returns a string, a flag indicating string has multiple lines,
        and a flag indicating that the formatted output has more than one line.
        """
        # Strip braces from parameter values, replace SI units
        # Compute full size
        newpar = []
        full_size = 0
        for name, value in params:
            value = self.format_value(value)
            newpar.append((name, value))
            full_size += len(name) + 1 + len(value) + 1
        params = newpar

        # Do we need to split the line
        need_split = atcol + full_size > self.cfg["columns"]

        split = False
        line = ""
        fragment = ""
        first = True
        for name, value in params:
            # Handle SI prefixes
            value = si_replace(value)

            # Do we need to start a new line
            if indent + len(fragment) + len(name) + 1 + len(value) > self.cfg["columns"]:
                # New line
                if not first:
                    line += "\n"
                    split = True
                else:
                    first = False
                line += fragment
                fragment = ""

            # Space as separator
            if len(fragment) > 0:
                fragment += " "

            # Add parameter
            fragment += name + "=" + value

        # Flush last fragment
        if len(fragment) > 0:
            if not first:
                line += "\n"
                split = True
            line += fragment

        return line, need_split, split

    def indent(self, txt, indent):
        """Indent text by *indent* spaces."""
        istr = " " * indent
        return "\n".join([istr + line for line in txt.split("\n")])

    def format_subckt_params(self, params, lws):
        """Format subcircuit parameters.

        One parameter per line.
        """
        txt = ""
        first = True
        for name, value in params:
            value = si_replace(value)
            if not first:
                txt += "\n"
            first = False
            txt += lws + "parameters " + name + "=" + value

        return txt

    def split_params(self, params, handle_m=False):
        """Split a list of parameter assignments into (name, value) pairs.

        Treats boolean parameters as <param>=1.
        If requested, renames m parameter.

        Delegates to spiceparser.params.split_params() with error handling.
        """
        try:
            return _split_params(params, handle_m=handle_m)
        except ValueError as e:
            raise ConverterError(str(e)) from e

    def remove_params(self, params, to_remove=None):
        """Remove all parameters listed in set *to_remove*.

        Can handle unsplit and split parameters.
        """
        if to_remove is None:
            to_remove = set()

        # Handle mixed unsplit/split params (original behavior)
        out = []
        for part in params:
            if isinstance(part, (list, tuple)):
                if part[0] in to_remove:
                    continue
            elif part in to_remove:
                continue
            out.append(part)
        return out

    def merge_vectors(self, params, vecnames=None):
        """Merge vector parameters into one string.

        Assumes *params* is a list of unsplit parameters.
        Delegates to spiceparser.params.merge_vector_params().
        """
        if vecnames is None:
            vecnames = set()
        return merge_vector_params(params, vecnames)

    def process_expressions(self, params):
        """Process expressions, replace temper -> $temp.

        Delegates to spiceparser.params.process_expressions().
        """
        return _process_expressions(params)

    def process_instance_params(self, params, insttype, handle_m=False):
        """Process instance parameters.

        Merges vectors, removes unneeded parameters.
        If requested, renames m parameter.

        Returns split parameters.
        """
        # Merge vector parameters
        vecnames = self.cfg["merge_vector_instance_params"].get(insttype, None)
        if vecnames is not None:
            params = self.merge_vectors(params, vecnames)

        # Split parameters, default boolean parameters, handle m
        psplit = self.split_params(params, handle_m=handle_m)

        # Remove unneeded parameters
        to_remove = self.cfg["remove_instance_params"].get(insttype, None)
        if to_remove is not None:
            psplit = self.remove_params(psplit, to_remove)

        # Process expressions
        psplit = self.process_expressions(psplit)

        return psplit

    def process_terminals(self, terminals):
        """Process terminals for VACASK output.

        Replaces invalid VACASK identifier characters:
        - ! -> _
        - [N] -> _N_ (array indices not allowed in VACASK identifiers)

        Delegates to spiceparser.params.process_terminals().
        """
        return _process_terminals(terminals)
