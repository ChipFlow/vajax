from .exc import ConverterError
from .generators import traverse
from .patterns import *


class OutputMixin:
    def proces_params(self, line):
        """
        Splits .param/.params, remove curly braces around expressions.
        """
        parts = line.split(" ")
        pairs = [p.split("=", 1) for p in parts[1:]]
        npairs = []
        for n, v in pairs:
            if v[0] == "{":
                v = v[1:-1]
            npairs.append((n, v))

        return parts[0] + " " + (" ".join([p + "=" + v for p, v in npairs]))

    def vacask_file(self):
        """
        Returns VACASK file as a list of lines.
        """
        deck = self.data["deck"]
        out = []
        first = True
        in_sub = None
        target_depth = self.cfg.get("output_depth", None)
        for history, line, depth, in_control_block in traverse(deck, depth=target_depth):
            # Skip control block
            if in_control_block:
                continue

            lnum, lws, l, eolc, annot = line

            # Special handling for title
            if first:
                first = False
                if self.data["is_toplevel"]:
                    out.append(self.data["title"])
                    if "signature" in self.cfg:
                        out.append(self.cfg["signature"])
                    # Add ground definition for VACASK
                    out.append("ground 0")
                    out.append("")
                    continue
                else:
                    if "signature" in self.cfg:
                        out.append(self.cfg["signature"])

            # Handle various lines
            if len(l) == 0:
                # Empty line
                out.append(lws + l)
            elif l.startswith("*"):
                # Comment
                out.append(lws + "//" + l)
            elif pat_cidotmodel.match(l):
                # Model, output if it is used or output is forced
                if (
                    self.cfg.get("all_models", False)
                    or len(self.data["model_usage"].get((annot["name"], in_sub), set())) > 0
                ):
                    out.append(self.process_model(lws, l, eolc, annot, in_sub))
            elif pat_cidotinclude.match(l):
                # Include
                if target_depth is None or depth < target_depth:
                    # Not at the deepest level yet,
                    # output a comment containing original .include
                    out.append(lws + "// " + l)
                else:
                    name, section, subdeck = eolc
                    out.append(lws + 'include "' + name + '"')
            elif pat_cidotlib.match(l):
                # Lib
                name, section, subdeck = eolc
                if name is None:
                    # Section start marker
                    out.append(lws + "section " + section)
                else:
                    # Library section include
                    if target_depth is None or depth < target_depth:
                        # Not at the deepest level yet,
                        # output a comment containing original .lib
                        out.append(lws + "// " + l)
                    else:
                        name, section, subdeck = eolc
                        out.append(lws + 'include "' + name + '" section=' + section)
            elif pat_cidotendl.match(l):
                # End of section marker
                out.append(lws + "endsection")
            elif pat_cidotsubckt.match(l):
                # Subcircuit start
                name = l.split(" ", 2)[1]
                in_sub = name
                orig_name = annot["origline"].split(" ", 2)[1]

                if self.cfg.get("original_case_subckt", False):
                    output_name = orig_name
                else:
                    output_name = name

                if self.debug > 0:
                    print((" " * self.dbgindent) + "toplevel subckt:", output_name)

                terminals, params = self.data["subckts"][name]
                vcline = lws + "subckt " + output_name + "("
                vcline += " ".join(terminals)
                vcline += ")"
                out.append(vcline)
                if len(params) > 0:
                    out.append(self.format_subckt_params(params, lws))
            elif pat_cidotends.match(l):
                in_sub = None
                out.append(lws + "ends")
            elif pat_cidotparams.match(l):
                # Remove curly braces
                txt = self.proces_params(l)
                txt = txt.replace(".params", "parameters")
                out.append(lws + txt)
            elif pat_cidotparam.match(l):
                # Remove curly braces
                txt = self.proces_params(l)
                txt = txt.replace(".param", "parameters")
                out.append(lws + txt)
            elif pat_cidotif.match(l):
                txt = l.replace(".if", "@if")
                out.append(lws + txt)
            elif pat_cidotelseif.match(l):
                txt = l.replace(".elseif", "@elseif")
                out.append(lws + txt)
            elif pat_cidotelse.match(l):
                txt = l.replace(".else", "@else")
                out.append(lws + txt)
            elif pat_cidotendif.match(l):
                txt = l.replace(".endif", "@end")
                out.append(lws + txt)
            elif pat_cidotend.match(l):
                # Done.
                break
            else:
                # Instance
                if l[0] >= "a" and l[0] <= "z":
                    method = getattr(self, "process_instance_" + l[0], None)
                    if method is None:
                        raise ConverterError(
                            "Dont' know how to process instances of type '" + l[0] + "'.",
                            history,
                            lnum,
                        )
                    try:
                        txt = method(lws, l, eolc, annot, in_sub)
                    except ConverterError as e:
                        raise ConverterError(str(e), history, lnum)
                    out.append(txt)

        if self.data["is_toplevel"]:
            # Dump load statements and default models.
            # This will happen only in the toplevel
            m = self.load_statements()
            if len(m) > 0:
                out.append("")
                out.append("// Modules")
                out.extend(m)

            m = self.default_models()
            if len(m) > 0:
                out.append("")
                out.append("// Default models")
                out.extend(m)

        return out
