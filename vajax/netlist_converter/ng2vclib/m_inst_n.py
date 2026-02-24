from .exc import ConverterError


class InstanceNMixin:
    def process_instance_n(self, lws, line, eol, annot, in_sub):
        """
        Process N instance (OSDI device).
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        if model is None:
            raise ConverterError(line + "\nModel not found.")

        terminals = self.process_terminals(parts[:mod_index])
        params = parts[(mod_index + 1) :]

        # Process parameters
        psplit = self.process_instance_params(params, "n", handle_m=True)

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
