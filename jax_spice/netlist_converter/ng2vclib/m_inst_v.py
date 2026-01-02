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
        name = annot["name"]
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

            # Check for PWL pattern (pwl(...) or pwl( broken by split)
            if rest_str.lower().startswith("pwl(") or rest_str.lower().startswith("pwl "):
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
            pwl_str_clean = re.sub(r'(\d+\.?\d*[eE]?[+-]?\d*)[sS]', r'\1', pwl_str)
            pwl_str_clean = re.sub(r'(\d+\.?\d*[eE]?[+-]?\d*)[vV]', r'\1', pwl_str_clean)
            tokens = pwl_str_clean.split()
            # Return flat list of alternating time, value pairs
            return "[" + ", ".join(tokens) + "]"

        # Spaces were removed - need to parse more carefully
        # Pattern: time value pairs where time ends in 's' and value ends in 'v'
        # Match pattern: (number)s(number)v repeated
        number_pattern = r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'
        pair_pattern = f'({number_pattern})[sS]({number_pattern})[vV]'

        values = []
        for match in re.finditer(pair_pattern, pwl_str):
            t = match.group(1)
            v = match.group(2)
            values.extend([t, v])

        return "[" + ", ".join(values) + "]"
