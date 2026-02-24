from .exc import ConverterError


class InstanceQMixin:
    def process_instance_q(self, lws, line, eol, annot, in_sub):
        """
        Process Q instance (bjt).

        qname c b e [s [tj]] <model> [<p1=value1> <p2=value2> ...]
        """
        annot["name"]
        parts = annot["words"]
        mod_index = annot["mod_index"]
        model = annot["mod_name"]

        terminals = self.process_terminals(parts[:mod_index])

        if model is None:
            # No model specified
            raise ConverterError("Model not specified.")
        else:
            # Have model
            if mod_index >= 3 and mod_index <= 5:
                model = parts[mod_index]
                params = parts[(mod_index + 1) :]
            else:
                # Don't know how to handle
                raise ConverterError("Cannot handle model at position " + str(mod_index + 1) + ".")

        # Process parameters
        psplit = self.process_instance_params(params, "q", handle_m=True)

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
                txt += fmted

        return txt
