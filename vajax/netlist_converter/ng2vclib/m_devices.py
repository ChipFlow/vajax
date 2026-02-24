class DevicesMixin:
    def load_statements(self):
        """
        Generate load statements.
        """
        # Scan all used models, collect osdi files.
        files = []
        for (name, in_sub), in_sub_use_set in self.data["model_usage"].items():
            # Look up model definition
            builtin, model_type, family, level, version, _ = self.data["models"][in_sub][name]
            if builtin:
                # Look up family
                file, module, _ = self.cfg["family_map"][family, level, version]
                # Add load
                files.append(file)

        # Add pre_osdi loaded files
        for osdi_file in self.data["osdi_loads"]:
            files.append(osdi_file)

        # Add default models.
        for inst_letter in self.data["default_models_needed"]:
            file, module = self.cfg["default_models"][inst_letter]
            # Add load
            files.append(file)

        # Generate text
        return ['load"' + f + '"' for f in files]

    def default_models(self):
        """
        Generate default model statements.
        """
        model_lines = []
        for inst_letter in self.data["default_models_needed"]:
            file, module = self.cfg["default_models"][inst_letter]
            # Add load
            model_lines.append((inst_letter, module))

        # Generate text for OSDI-based default models
        result = [
            "model " + self.cfg["default_model_prefix"] + inst_letter + " " + module
            for inst_letter, module in model_lines
        ]

        # Add builtin model declarations (vsource, isource, etc.)
        if "builtin_models_needed" in self.data:
            for model in self.data["builtin_models_needed"]:
                result.append(f"model {model} {model}")

        return result
