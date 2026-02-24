import os

from .exc import ConverterError
from .patterns import *


def remove_paired_parentheses_spaces(l):
    """
    Remove spaces from within paired parentheses.
    """
    stack = []
    result = ""
    for i, char in enumerate(l):
        if char == "(":
            stack.append(i)
            continue
        elif char == ")":
            start = stack.pop()
            if len(stack) == 0:
                inner = l[start:i].replace(" ", "")
                result += inner + char
            continue

        if len(stack) == 0:
            result += char
            continue

    return result


class FileLoaderMixin:
    def preprocess_line(self, line):
        """
        Preprocesses netlist line, ckeck for toplevel circuit.

        All operations here work the same even if the line was processed
        with lower() before calling this function.
        """
        # Preprocess netlist part (outside control block)
        l = line

        # Strip trailing whitespace from core line
        l = l.rstrip()
        # Replace tabs with spaces
        l = l.replace("\t", " ")
        # Replace sequences of spaces with single space
        l = pat_spaceseq.sub(" ", l)
        # Remove sequences of spaces before and after =
        l = pat_spaceequal.sub("=", l)
        # Replace pairs of single quotes with curly braces
        l = pat_singlequotes.sub(r"{\1}", l)
        # Remove spaces from strings in curly braces, remove curly braces
        l = pat_cbracepair.sub(lambda m: "{" + m.group(1).replace(" ", "") + "}", l)

        # For .model lines, remove parentheses around parameters
        # .model name type (...)
        if pat_cidotmodel.search(l):
            mp = l.split(" ", 3)
            if len(mp) > 3 and mp[3].startswith("(") and mp[3].endswith(")"):
                stripped = mp[3][1:-1].strip()
                l = " ".join(mp[:3] + [stripped])

        # Remove spaces from within paired parentheses
        l = remove_paired_parentheses_spaces(l)

        # Check if this is a toplevel circuit
        if self.cfg["as_toplevel"] == "auto":
            if pat_cidotend_line.match(l):
                self.data["is_toplevel"] = True

        return l

    def find_file(self, filename):
        """
        Looks for a file in current directory and in the sourcepath.
        If found, returns its path.
        If not found, returns None.
        """
        if os.path.isfile(filename):
            return filename

        if not os.path.isabs(filename):
            for p in self.cfg["sourcepath"]:
                full_path = os.path.join(p, filename)
                if os.path.isfile(full_path):
                    return full_path

        return None

    def read_file(self, filename, section=None, depth=0, inside_control=False):
        """
        Reads a file.

        If *section* is not None returns only the given section.
        If section is not found, raises an exception.
        *depth* is the inclusion deptf of the file.
        *inside_control* specifies whether the file is inside a control block.

        Returns a tuple with two members if depth>0:
        * *inside_control* status after the end of file is reached
        * deck
        * canonical path to the file that was read

        Otherwise returns only the deck.

        Deck is a tuple with the following members
        * file name
        * section name (for .lib) or None (for .include)
        * a list of tuples (corresponding to lines) with the following members
          * line number
          * leading whitespace
          * core line
          * eol comment
          * annotations dictionary

        If recursive_read is enabled in the configuration and a line is an
        .include or a .lib directive the eol comment entry is a tuple holding
        * file name
        * section name (None for .include)
        * a deck

        Preprocesses lines, removes trailing whitespace from core line.
        Merges continued lines (+).
        Converts all parts to lowercase, except for the control block and
        .lib/.include directives.
        Converts pairs of single quotes around expressions into curly braces.
        Converts tabs to spaces. Removes excessive spaces.
        Separates control block.
        """
        # Initialize toplevel circuit status
        self.data["is_toplevel"] = self.cfg["as_toplevel"] == "yes"

        # Read input file, strip CR/LF
        fp = self.find_file(filename)
        if fp is None:
            raise ConverterError("File " + filename + " not found")
        try:
            with open(fp, errors="ignore") as file:
                lines = [line.rstrip("\r\n") for line in file]
        except (OSError, IOError, UnicodeDecodeError) as e:
            raise ConverterError(f"Failed to open {fp}: {e}")

        # Canonical path
        fp = os.path.realpath(fp)

        # Check if it needs patching
        for patchfile, pl in self.cfg.get("patch", {}).items():
            if fp.endswith(patchfile):
                # Do the patching
                nlines = []
                for line in lines:
                    for porig, pchange in pl:
                        if line.startswith(porig):
                            line = pchange
                    nlines.append(line)
                lines = nlines

        # Extract title
        if depth == 0:
            self.data["title"] = lines[0]

        # Split into leading spaces, core, and trailing eol comment
        # Merge lines that start with continuation character.
        # Line comments between merged lines are dropped.
        # End of line comment on a line to which a line is merged are dropped.
        nlines = []
        comments = []
        collect = section is None
        for lnum, l in enumerate(lines):
            if len(l) == 0:
                # Empty line, add to comments
                if collect:
                    comments.append((lnum, "", l, "", {}))
                continue

            # Separate leading whitespace
            match = pat_leadspace.search(l)
            if match:
                lws = l[: match.end()]
                l = l[match.end() :]
            else:
                lws = ""

            # Line comment, keep case
            if len(l) > 0 and l[0] == "*":
                if collect:
                    comments.append((lnum, lws, l, "", {}))
                continue

            # Is it a .lib or .include line
            # Look for match at beginning of string (match())
            # Leading whitespace has been removed
            islib = pat_cidotlib.match(l)
            libmarker = None
            isinclude = pat_cidotinclude.match(l)
            isendl = pat_cidotendl.match(l)
            iscontrol = pat_cidotcontrol.match(l)
            isendc = pat_cidotendc.match(l)

            if islib:
                # Check for lib section start
                # Convert sequences of whitespace to single spaces, strip, and split
                lcomp = pat_spaceseq.sub(" ", l.strip()).split(" ")
                # Is it a section marker, i.e. 2 components
                # Section names do not contain whitespace
                if len(lcomp) == 2:
                    # This is a section marker
                    sec = lcomp[1]
                    libmarker = sec
                    if section is not None:
                        # Looking for section, Check section
                        if sec == section:
                            collect = True
                            # Do not store
                            continue
                    elif depth > 0:
                        # Not looking for section, depth>0
                        # Stray .lib section marker, probably included a .lib file
                        # Error
                        raise ConverterError(
                            filename + ", line " + str(lnum + 1) + ": stray section marker."
                        )
                    else:
                        # Not looking for section, depth is 0
                        # Store section marker (will do it later)
                        pass
            elif isendl:
                # Check for lib section end
                # Do not check section name, assume file is sane
                if section is not None:
                    # Stop collecting
                    collect = False
                    # Do not store
                    continue
                elif depth > 0:
                    # Not looking for a section, depth>0
                    # Stray .endl, error
                    raise ConverterError(
                        filename + ", line " + str(lnum + 1) + ": stray end of section marker."
                    )
                else:
                    # Not looking for section, depth is 0
                    # Store section marker (will do it later)
                    pass
            elif iscontrol:
                # Start of control block
                inside_control = True
            elif isendc:
                # End of control block
                inside_control = False

            # Not in collect mode, continue
            if not collect:
                continue

            # Assume no eol comment
            eolc = ""

            if isinclude:
                # Recursively read included files
                # Extract file name
                # Skip .include, strip whitespace
                s = l[8:].strip()
                # Unquote
                if s.startswith("'") and s.endswith("'") or s.startswith('"') and s.endswith('"'):
                    s = s[1:-1]
                # Load include file
                rd = self.cfg.get("read_depth", None)
                if rd is None or depth < rd:
                    inside_control, subdeck, _ = self.read_file(
                        s, depth=depth + 1, inside_control=inside_control
                    )
                    eolc = subdeck
                else:
                    # Not going further, just store extracted filename
                    eolc = (s, None, None)
            elif islib:
                # Handle .lib recursive read and .lib marker
                if libmarker is not None:
                    # This is a section marker, store its data
                    eolc = (None, libmarker, None)
                else:
                    # This is a .lib include statement
                    # Extract file name and section
                    # Skip .lib, strip whitespace
                    s = l[4:].strip()
                    # Find first whitespace from right to left
                    sndx = s.rfind(" ")
                    if sndx < 0:
                        # Not found, must be a stray section marker
                        # This is an error if this is not the top file
                        if depth > 0:
                            raise ConverterError(
                                filename + ", line " + str(lnum + 1) + ": stray section marker."
                            )
                    # Extract file name, strip whitespace
                    lfname = s[:sndx].strip()
                    # Unquote
                    if (
                        lfname.startswith("'")
                        and lfname.endswith("'")
                        or lfname.startswith('"')
                        and lfname.endswith('"')
                    ):
                        lfname = lfname[1:-1]
                    # Get section name
                    sname = s[sndx:].strip()
                    # Load lib file
                    rd = self.cfg.get("read_depth", None)
                    if rd is None or depth < rd:
                        inside_control, subdeck, _ = self.read_file(
                            lfname, section=sname, depth=depth + 1, inside_control=inside_control
                        )
                        eolc = subdeck
                    else:
                        # Not going further, just store extracted filename and section name
                        eolc = (lfname, sname, None)
            elif not inside_control:
                # Not .lib, .include, or control block
                # Separate trailing eol comment
                match = pat_eolcomment.search(l)
                if match:
                    eolc = "//" + l[match.start() :]
                    l = l[: match.start()]

            # Conclude processing
            # Convert to lowercase every non-comment outside control block
            if len(l) == 0:
                # Empty core line, add to comments
                if collect:
                    comments.append((lnum, lws, l, eolc, {"isnl": False}))
                continue
            elif l[0] == "+":
                # Continuation line
                # Do we have a previous line
                if len(nlines) <= 0:
                    raise ConverterError(
                        filename
                        + ", line "
                        + str(lnum + 1)
                        + ": cannot continue a line without a previous line."
                    )
                # Drop collected comments
                comments = []
                if collect:
                    # Remove end-of-line comment from last line, merge with continuation line and its eol comment
                    prev_lnum, prev_lws, prev_line, _, _ = nlines[-1]
                    # Add extra space
                    nlines[-1] = (
                        prev_lnum,
                        prev_lws,
                        prev_line + " " + l[1:],
                        eolc,
                        {"isnl": not inside_control},
                    )
            else:
                # Ordinary line
                # Flush comments
                if collect:
                    nlines.extend(comments)
                comments = []
                if collect:
                    # Convert to lowercase if outside control block
                    nlines.append((lnum, lws, l, eolc, {"isnl": not inside_control}))

        # Flush trailing comments
        nlines.extend(comments)

        # Go through all lines, preprocess lines outside control block
        lines = []
        for lnum, lws, line, eolc, data in nlines:
            if data.get("isnl", False):
                # Store raw line before any preprocessing (for PULSE/PWL parsing)
                data["rawline"] = line

                # Netlist lines are converted to lowercase and preprocessed
                # Preprocess first
                line = self.preprocess_line(line)

                # Store original case (after preprocessing)
                data["origline"] = line

                # To lowercase
                line = line.lower()

            lines.append((lnum, lws, line, eolc, data))

        # Lines is now a list of tuples of the form
        # (line number, leading whitespace, core line, trailing eol comment, extra data)

        if depth > 0:
            return (inside_control, (filename, section, lines), fp)
        else:
            if inside_control:
                raise ConverterError("Unterminated control block found.")

            return (inside_control, (filename, section, lines), fp)
