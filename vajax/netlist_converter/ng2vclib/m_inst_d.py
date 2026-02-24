from .exc import ConverterError


class InstanceDMixin:
    def process_instance_d(self, lws, line, eol, annot, in_sub):
        """
        Process D instance (diode).

        dname p n <model> [<p1=value1> <p2=value2> ...]
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        terminals = self.process_terminals(parts[:2])

        if model is None:
            # No model specified
            raise ConverterError("Model not specified.")
        else:
            # Have model
            if mod_index == 2:
                # Third entry, immediately after terminals
                model = parts[mod_index]
                params = parts[3:]
            else:
                # Don't know how to handle
                raise ConverterError("Cannot handle model at position " + str(mod_index + 1) + ".")

        # Process parameters
        psplit = self.process_instance_params(params, "d", handle_m=True)

        txt = (
            lws
            + annot["output_name"]
            + " ("
            + (" ".join(terminals))
            + ") "
            + annot["output_mod_name"]
            + " "
        )

        if len(psplit) > 0:
            fmted, need_split, split = self.format_params(psplit, len(txt))
            if need_split or split:
                fmted = self.indent(fmted, len(lws) + 2)
                txt += "(" + eol + "\n" + fmted
                txt += "\n" + lws + ")"
            else:
                txt += " " + fmted

        return txt
