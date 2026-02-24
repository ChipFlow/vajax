from .exc import ConverterError


class InstancePassiveMixin:
    def process_instance_r(self, lws, line, eol, annot, in_sub):
        """
        Process R instance (resistor).

        rname p n <r=value>|<value> [<p1=value1> <p2=value2> ...]
        rname p n <model> [<p1=value1> <p2=value2> ...]
        rname p n <value> <model> [<p1=value1> <p2=value2> ...]
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        terminals = self.process_terminals(parts[:2])

        if model is None:
            # No model specified
            model = self.cfg["default_model_prefix"] + "r"
            self.data["default_models_needed"].add("r")

            # Check if part 3 is not a parameter assignment
            if "=" not in parts[2]:
                # Part 3 is the resistance, the rest are parameter assignments
                params = parts[3:]

                # Add value as first parameter assignment
                psplit = [("r", self.format_value(parts[2]))]
            else:
                # Part 3 is a parameter assignment
                params = parts[2:]

                # No parameter assignments yet
                psplit = []

        else:
            # Have model
            if mod_index == 2:
                # Thrrd entry, immediately after terminals, no value
                psplit = []
                model = annot["output_mod_name"]
                params = parts[3:]
            elif mod_index == 3:
                # Model is 4th entry, 3rd entry must be a value
                psplit = [("r", self.format_value(parts[2]))]
                model = annot["output_mod_name"]
                params = parts[4:]
            else:
                # Don't know how to handle
                raise ConverterError("Cannot handle model at position " + str(mod_index + 1) + ".")

        # Process parameters - extend psplit rather than replace
        psplit.extend(self.process_instance_params(params, "r", handle_m=True))

        txt = lws + annot["output_name"] + " (" + (" ".join(terminals)) + ") " + model + " "

        if len(psplit) > 0:
            fmted, need_split, split = self.format_params(psplit, len(txt))
            if need_split or split:
                fmted = self.indent(fmted, len(lws) + 2)
                txt += "(" + eol + "\n" + fmted
                txt += "\n" + lws + ")"
            else:
                txt += " " + fmted

        return txt

    def process_instance_c(self, lws, line, eol, annot, in_sub):
        """
        Process C instance (capacitor).

        cname p n <value> [<p1=value1> <p2=value2> ...]
        cname p n <model> [<p1=value1> <p2=value2> ...]
        cname p n <value> <model> [<p1=value1> <p2=value2> ...]

        Removes ic parameter.
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        terminals = self.process_terminals(parts[:2])

        if model is None:
            # No model specified
            model = self.cfg["default_model_prefix"] + "c"
            self.data["default_models_needed"].add("c")

            # Check if part 3 is not a parameter assignment (like resistor handler)
            if len(parts) > 2 and "=" not in parts[2]:
                # Part 3 is the capacitance, the rest are parameter assignments
                params = parts[3:]

                # Add value as first parameter assignment
                psplit = [("c", self.format_value(parts[2]))]
            else:
                # Part 3 is a parameter assignment
                params = parts[2:]

                # No parameter assignments yet
                psplit = []
        else:
            # Have model
            if mod_index == 2:
                # Third entry, immediately after terminals, no value
                psplit = []
                model = annot["output_mod_name"]
                params = parts[3:]
            elif mod_index == 3:
                # Model is 4th entry, 3rd entry must be a value
                psplit = [("c", self.format_value(parts[2]))]
                model = annot["output_mod_name"]
                params = parts[4:]
            else:
                # Don't know how to handle
                raise ConverterError("Cannot handle model at position " + str(mod_index + 1) + ".")

        # Process parameters - extend psplit rather than replace
        psplit.extend(self.process_instance_params(params, "c", handle_m=True))

        txt = lws + annot["output_name"] + " (" + (" ".join(terminals)) + ") " + model + " "

        if len(psplit) > 0:
            fmted, need_split, split = self.format_params(psplit, len(txt))
            if need_split or split:
                fmted = self.indent(fmted, len(lws) + 2)
                txt += "(" + eol + "\n" + fmted
                txt += "\n" + lws + ")"
            else:
                txt += " " + fmted

        return txt

    def process_instance_l(self, lws, line, eol, annot, in_sub):
        """
        Process L instance (inductor).

        lname p n <value> [<p1=value1> <p2=value2> ...]
        lname p n <model> [<p1=value1> <p2=value2> ...]
        lname p n <value> <model> [<p1=value1> <p2=value2> ...]

        Removes ic parameter.
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        terminals = self.process_terminals(parts[:2])

        if model is None:
            # No model specified
            model = self.cfg["default_model_prefix"] + "l"
            self.data["default_models_needed"].add("l")

            # Part 3 is a parameter assignment
            params = parts[2:]

            # No parameter assignments yet
            psplit = []
        else:
            # Have model
            if mod_index == 2:
                # Third entry, immediately after terminals, no value
                psplit = []
                model = annot["output_mod_name"]
                params = parts[3:]
            elif mod_index == 3:
                # Model is 4th entry, 3rd entry must be a value
                psplit = [("l", self.format_value(parts[2]))]
                model = annot["output_mod_name"]
                params = parts[4:]
            else:
                # Don't know how to handle
                raise ConverterError("Cannot handle model at position " + str(mod_index + 1) + ".")

        # Process parameters - extend psplit rather than replace
        psplit.extend(self.process_instance_params(params, "l", handle_m=True))

        txt = lws + annot["output_name"] + " (" + (" ".join(terminals)) + ") " + model + " "

        if len(psplit) > 0:
            fmted, need_split, split = self.format_params(psplit, len(txt))
            if need_split or split:
                fmted = self.indent(fmted, len(lws) + 2)
                txt += "(" + eol + "\n" + fmted
                txt += "\n" + lws + ")"
            else:
                txt += " " + fmted

        return txt
