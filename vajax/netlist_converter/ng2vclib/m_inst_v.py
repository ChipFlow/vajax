import re


class InstanceVMixin:
    def process_instance_v(self, lws, line, eol, annot, in_sub):
        """
        Process V instance (voltage source).

        Ngspice format:
            vname n+ n- dc=value
            vname n+ n- value
            vname n+ n- pwl(t1 v1 t2 v2 ...)
            vname n+ n- pulse(v1 v2 td tr tf pw per)

        VACASK format:
            vname (n+ n-) vsource dc=value
            vname (n+ n-) vsource type="pwl" wave=[t1, v1, t2, v2, ...]
        """
        annot["name"]
        parts = annot["words"]

        # Track that vsource builtin model is needed
        if "builtin_models_needed" not in self.data:
            self.data["builtin_models_needed"] = set()
        self.data["builtin_models_needed"].add("vsource")

        terminals = self.process_terminals(parts[:2])

        # Rest of line after terminals - reconstruct from original line
        # to preserve PWL parentheses content
        rest = parts[2:] if len(parts) > 2 else []

        model = "vsource"
        params = []

        if len(rest) == 0:
            # No parameters - default to dc=0
            params.append(("dc", "0"))
        else:
            # Reconstruct the rest of the line
            rest_str = " ".join(rest)

            # Handle "dc VALUE" prefix before transient source
            # ngspice format: "dc 0 pulse 0 1 1u 1u 1u 1m 2m"
            dc_value = None
            rest_str_lower = rest_str.lower()
            if rest_str_lower.startswith("dc="):
                # dc=value format
                dc_end = rest_str.find(" ", 3)
                if dc_end == -1:
                    dc_value = rest_str[3:]
                    rest_str = ""
                else:
                    dc_value = rest_str[3:dc_end]
                    rest_str = rest_str[dc_end:].strip()
                    rest_str_lower = rest_str.lower()
            elif rest_str_lower.startswith("dc "):
                # dc value format
                parts_dc = rest_str.split(None, 2)
                if len(parts_dc) >= 2:
                    dc_value = parts_dc[1]
                    rest_str = parts_dc[2] if len(parts_dc) > 2 else ""
                    rest_str_lower = rest_str.lower()

            # If we extracted a DC value and there's no transient source,
            # add the DC value to params now
            if dc_value is not None and not rest_str:
                params.append(("dc", self.format_value(dc_value)))

            # Check for PWL pattern (pwl(...) or pwl( broken by split)
            if rest_str_lower.startswith("pwl(") or rest_str_lower.startswith("pwl "):
                # Extract PWL content
                pwl_content = rest_str
                if pwl_content.lower().startswith("pwl("):
                    pwl_content = pwl_content[4:]  # Remove "pwl("
                elif pwl_content.lower().startswith("pwl"):
                    pwl_content = pwl_content[3:].strip()
                    if pwl_content.startswith("("):
                        pwl_content = pwl_content[1:]

                # Remove trailing )
                if pwl_content.endswith(")"):
                    pwl_content = pwl_content[:-1]

                pwl_data = self.parse_pwl(pwl_content)
                params.append(("type", '"pwl"'))
                params.append(("wave", pwl_data))
            elif (
                rest_str_lower.startswith("pulse(")
                or rest_str_lower.startswith("pulse ")
                or rest_str_lower.startswith("pulse")
            ):
                # PULSE source: PULSE(v1 v2 td tr tf pw per) or PULSE v1 v2 td tr tf pw per
                # VACASK format: type="pulse" val0=v1 val1=v2 delay=td rise=tr fall=tf width=pw period=per
                #
                # Use raw line (before preprocessing) since preprocess removes spaces inside parens
                raw_line = annot.get("rawline", annot.get("origline", ""))
                pulse_match = re.search(r"pulse\s*\(\s*([^)]+)\s*\)", raw_line, re.IGNORECASE)
                if pulse_match:
                    pulse_content = pulse_match.group(1).strip()
                else:
                    # Fallback to preprocessed content - extract after "pulse"
                    pulse_content = rest_str
                    if pulse_content.lower().startswith("pulse("):
                        pulse_content = pulse_content[6:]
                    elif pulse_content.lower().startswith("pulse"):
                        pulse_content = pulse_content[5:].strip()
                        if pulse_content.startswith("("):
                            pulse_content = pulse_content[1:]
                    if pulse_content.endswith(")"):
                        pulse_content = pulse_content[:-1]

                pulse_params = self.parse_pulse(pulse_content)
                params.append(("type", '"pulse"'))
                params.extend(pulse_params)
            elif rest_str.lower().startswith("dc="):
                params.append(("dc", self.format_value(rest_str[3:])))
            elif rest_str.lower().startswith("dc "):
                params.append(("dc", self.format_value(rest_str[3:].strip())))
            elif len(rest) == 1 and "=" not in rest[0]:
                # Single value - treat as DC
                params.append(("dc", self.format_value(rest[0])))
            elif "=" in rest[0]:
                # Parameter assignment
                p, v = rest[0].split("=", 1)
                params.append((p.strip(), self.format_value(v.strip())))
            elif dc_value is not None:
                # Already handled DC value above - nothing more to do
                pass
            else:
                # Treat first as DC value
                params.append(("dc", self.format_value(rest[0])))

        txt = lws + annot["output_name"] + " (" + " ".join(terminals) + ") " + model

        if len(params) > 0:
            param_strs = []
            for pn, pv in params:
                param_strs.append(f"{pn}={pv}")
            txt += " " + " ".join(param_strs)

        return txt

    def parse_pwl(self, pwl_str):
        """Parse PWL data string and return VACASK format.

        Returns a flat list [t1, v1, t2, v2, ...] for VACASK wave parameter.
        """
        # The preprocessor removes spaces from inside parentheses,
        # so input may be like: "0.0s0.0v1e-08s0.0v1.02e-08s1.3v"
        # We need to split on 's' and 'v' unit suffixes.
        #
        # Pattern: number followed by 's' (time), then number followed by 'v' (voltage)
        # The number pattern is: optional sign, digits, optional decimal, optional exponent

        # First, try splitting by whitespace (in case spaces are preserved)
        tokens = pwl_str.split()
        if len(tokens) > 1:
            # Spaces were preserved - original behavior
            pwl_str_clean = re.sub(r"(\d+\.?\d*[eE]?[+-]?\d*)[sS]", r"\1", pwl_str)
            pwl_str_clean = re.sub(r"(\d+\.?\d*[eE]?[+-]?\d*)[vV]", r"\1", pwl_str_clean)
            tokens = pwl_str_clean.split()
            # Return flat list of alternating time, value pairs
            return "[" + ", ".join(tokens) + "]"

        # Spaces were removed - need to parse more carefully
        # Pattern: time value pairs where time ends in 's' and value ends in 'v'
        # Match pattern: (number)s(number)v repeated
        number_pattern = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
        pair_pattern = f"({number_pattern})[sS]({number_pattern})[vV]"

        values = []
        for match in re.finditer(pair_pattern, pwl_str):
            t = match.group(1)
            v = match.group(2)
            values.extend([t, v])

        return "[" + ", ".join(values) + "]"

    def parse_pulse(self, pulse_str):
        """Parse PULSE data string and return VACASK format parameters.

        SPICE format: PULSE(v1 v2 [td [tr [tf [pw [per]]]]])
        - v1: Initial value
        - v2: Pulsed value
        - td: Delay time (default 0)
        - tr: Rise time (default 0)
        - tf: Fall time (default 0)
        - pw: Pulse width (default infinity)
        - per: Period (default infinity)

        VACASK format: type="pulse" val0=v1 val1=v2 delay=td rise=tr fall=tf width=pw period=per

        Note: Preprocessing may remove spaces inside parentheses, so we need to
        handle both "5 -1 0.05ms" and "5-10.05ms100ns" formats.
        """
        # First try splitting by whitespace
        tokens = pulse_str.split()

        if len(tokens) > 1:
            # Spaces preserved - use simple split
            pass
        else:
            # Spaces removed - need to parse by unit suffixes and sign changes
            # Pattern: number (with optional SI suffix) repeated
            # Examples of numbers: 5, -1, 0.05ms, 100ns, 1e-9, 1.5e-6s
            tokens = self._parse_concatenated_values(pulse_str)

        params = []

        # Map SPICE pulse params to VACASK names
        param_names = ["val0", "val1", "delay", "rise", "fall", "width", "period"]

        for i, token in enumerate(tokens):
            if i < len(param_names):
                params.append((param_names[i], self.format_value(token)))

        return params

    def _parse_concatenated_values(self, s):
        """Parse a string of concatenated numeric values.

        Input: "5-10.05ms100ns100ns0.1ms0.2ms"
        Output: ["5", "-1", "0.05ms", "100ns", "100ns", "0.1ms", "0.2ms"]

        Strategy: Since preprocessing removes spaces, we need to find boundaries.
        Key insight for PULSE: first two values are voltages (no units required),
        remaining values are times (usually have units like ms, ns, us, s).

        We look for unit suffixes as boundaries. When a unit is found, it marks
        the end of one value. A new value starts at a digit or sign after a unit.
        """
        # SI unit suffixes (longer first for correct matching)
        units = [
            "meg",
            "mil",
            "ms",
            "us",
            "ns",
            "ps",
            "fs",
            "f",
            "p",
            "n",
            "u",
            "m",
            "k",
            "g",
            "t",
            "s",
            "v",
        ]

        # First pass: find all positions where units appear
        # Then work backwards to identify number boundaries
        tokens = []
        remaining = s

        while remaining:
            # Find a number at the start
            # A number is: optional sign, digits, optional decimal, optional exponent
            num_match = re.match(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", remaining)
            if not num_match:
                break

            num = num_match.group(1)
            remaining = remaining[len(num) :]

            # Check if followed by a unit suffix
            unit = ""
            for u in units:
                if remaining.lower().startswith(u):
                    # Make sure unit is not followed by another letter (to avoid partial matches)
                    if len(remaining) == len(u) or not remaining[len(u)].isalpha():
                        unit = remaining[: len(u)]
                        remaining = remaining[len(u) :]
                        break

            tokens.append(num + unit)

            # If there's still content but no more matches, something went wrong
            if remaining and not remaining[0].lstrip("+-").replace(".", "", 1).isdigit():
                # Skip any non-numeric characters
                idx = 0
                while idx < len(remaining) and not (
                    remaining[idx].isdigit() or remaining[idx] in "+-"
                ):
                    idx += 1
                remaining = remaining[idx:]

        return tokens
