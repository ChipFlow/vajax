class ModelMixin:
    def process_model(self, lws, line, eol, annot, in_sub):
        """
        Process a model line.
        """
        # Model
        name = line.split(" ")[1]
        if self.cfg.get("original_case_model", False):
            output_name = annot["origline"].split(" ")[1]
        else:
            output_name = name

        if in_sub is None and self.debug > 0:
            print((" " * self.dbgindent) + "toplevel model: ", output_name)

        builtin, mtype, family, level, version, params = self.data["models"][in_sub][name]

        paren = False
        if mtype in self.cfg["type_map"]:
            # Builtin model
            extra_params, family, _, _ = self.cfg["type_map"][mtype]
            osdifile, module, extra_family_params = self.cfg["family_map"][family, level, version]

            # Get names of parameters to remove
            vecnames = self.cfg["remove_model_params"].get(module, None)

            # Output
            vcline = "model " + output_name + " " + module
            if len(extra_params) > 0 or len(params) > 0:
                vcline += " ( " + eol
                paren = True
            else:
                vcline += " " + eol

            if len(extra_params) > 0:
                vcline += (
                    "\n"
                    + self.format_extra_params(extra_params, lws, 2)
                    + self.format_extra_params(extra_family_params, " ", 0)
                )
        else:
            # External model (osdi)

            # Get names of parameters to remove
            vecnames = self.cfg["remove_model_params"].get(mtype, None)

            # Output
            vcline = "model " + output_name + " " + mtype
            if len(params) > 0:
                vcline += " ( " + eol
                paren = True
            else:
                vcline += " " + eol

        if len(params) >= 0:
            # Remove parameters
            if vecnames is not None:
                params = self.remove_params(params, vecnames)

            fmted, _, _ = self.format_params(params, len(vcline))
            fmted = self.indent(fmted, len(lws) + 2)
            vcline += "\n" + fmted

        if paren:
            vcline += "\n" + lws + ")"

        return vcline
