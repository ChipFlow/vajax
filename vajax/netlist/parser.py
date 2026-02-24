"""VACASK netlist parser - simple recursive descent

Parses VACASK format netlists for circuit simulation.
Adapted from VACASK dflparser grammar.
Passes all VACASK test cases (vendor/VACASK/test/*.sim).
See docs/vacask_sim_format.md for details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from vajax.netlist.circuit import (
    AnalysisDirective,
    Circuit,
    ControlBlock,
    Instance,
    Model,
    OptionsDirective,
    PrintDirective,
    SaveDirective,
    Subcircuit,
    VarDirective,
)


@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int


class Lexer:
    """Simple lexer for VACASK netlist format"""

    KEYWORDS = {
        # Netlist structure
        "load",
        "include",
        "global",
        "ground",
        "model",
        "parameters",
        "subckt",
        "ends",
        "control",
        "endc",
        "embed",
        # Control section commands (from VACASK cmd.cpp)
        "options",
        "analysis",
        "save",
        "var",
        "print",
        "alter",
        "postprocess",
        "abort",
        "clear",
        "elaborate",
    }

    # Preprocessor directives (handled specially)
    # Note: @end is alias for @endif in some VACASK files
    DIRECTIVES = {"@if", "@elseif", "@else", "@endif", "@end"}

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []
        self._tokenize()

    def _tokenize(self):
        while self.pos < len(self.text):
            # Skip whitespace (but not newlines)
            if self.text[self.pos] in " \t":
                self.pos += 1
                self.col += 1
                continue

            # Newline
            if self.text[self.pos] == "\n":
                self.tokens.append(Token("NL", "\n", self.line, self.col))
                self.pos += 1
                self.line += 1
                self.col = 1
                continue

            if self.text[self.pos] == "\r":
                self.pos += 1
                continue

            # Line comment (//)
            if self.pos + 1 < len(self.text) and self.text[self.pos : self.pos + 2] == "//":
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
                continue

            # Block comment (/* ... */)
            if self.pos + 1 < len(self.text) and self.text[self.pos : self.pos + 2] == "/*":
                self.pos += 2
                self.col += 2
                while self.pos + 1 < len(self.text):
                    if self.text[self.pos : self.pos + 2] == "*/":
                        self.pos += 2
                        self.col += 2
                        break
                    if self.text[self.pos] == "\n":
                        self.line += 1
                        self.col = 1
                    else:
                        self.col += 1
                    self.pos += 1
                continue

            # @ directive (preprocessor)
            if self.text[self.pos] == "@":
                start = self.pos
                self.pos += 1
                # Collect directive name
                while self.pos < len(self.text) and self.text[self.pos].isalpha():
                    self.pos += 1
                word = self.text[start : self.pos]
                if word.lower() in ("@if", "@elseif", "@else", "@endif", "@end"):
                    self.tokens.append(Token("DIRECTIVE", word, self.line, self.col))
                else:
                    self.tokens.append(Token("NAME", word, self.line, self.col))
                self.col += self.pos - start
                continue

            # String
            if self.text[self.pos] == '"':
                start = self.pos
                self.pos += 1
                while self.pos < len(self.text) and self.text[self.pos] != '"':
                    self.pos += 1
                self.pos += 1  # closing quote
                self.tokens.append(
                    Token("STRING", self.text[start : self.pos], self.line, self.col)
                )
                self.col += self.pos - start
                continue

            # Special chars
            if self.text[self.pos] == "(":
                self.tokens.append(Token("LPAREN", "(", self.line, self.col))
                self.pos += 1
                self.col += 1
                continue

            if self.text[self.pos] == ")":
                self.tokens.append(Token("RPAREN", ")", self.line, self.col))
                self.pos += 1
                self.col += 1
                continue

            if self.text[self.pos] == "=":
                self.tokens.append(Token("EQ", "=", self.line, self.col))
                self.pos += 1
                self.col += 1
                # After =, collect the value (may contain expressions with parens)
                # Skip whitespace first
                while self.pos < len(self.text) and self.text[self.pos] in " \t":
                    self.pos += 1
                    self.col += 1
                if self.pos < len(self.text) and self.text[self.pos] == '"':
                    # String value - will be handled in next iteration
                    continue
                # Collect expression value, handling balanced parentheses and brackets
                start = self.pos
                paren_depth = 0
                bracket_depth = 0
                while self.pos < len(self.text):
                    ch = self.text[self.pos]
                    if ch == "(":
                        paren_depth += 1
                    elif ch == ")":
                        if paren_depth == 0:
                            break  # Unmatched ) - end of value
                        paren_depth -= 1
                    elif ch == "[":
                        bracket_depth += 1
                    elif ch == "]":
                        if bracket_depth == 0:
                            break  # Unmatched ] - end of value
                        bracket_depth -= 1
                    elif ch in " \t\n\r" and paren_depth == 0 and bracket_depth == 0:
                        break  # Whitespace outside parens/brackets - end of value
                    elif ch == "," and paren_depth == 0 and bracket_depth == 0:
                        break  # Comma outside brackets - end of value
                    self.pos += 1
                if self.pos > start:
                    val = self.text[start : self.pos]
                    self.tokens.append(Token("VALUE", val, self.line, self.col))
                    self.col += self.pos - start
                continue

            # Identifier or keyword or number
            if self.text[self.pos].isalnum() or self.text[self.pos] == "_":
                start = self.pos
                while self.pos < len(self.text) and (
                    self.text[self.pos].isalnum() or self.text[self.pos] in "_."
                ):
                    self.pos += 1
                word = self.text[start : self.pos]
                if word.lower() in self.KEYWORDS:
                    self.tokens.append(Token("KW_" + word.upper(), word, self.line, self.col))
                else:
                    self.tokens.append(Token("NAME", word, self.line, self.col))
                self.col += self.pos - start
                continue

            # Value (for param=value, can include special chars)
            # Skip for now, handle in parser
            self.pos += 1
            self.col += 1

        self.tokens.append(Token("EOF", "", self.line, self.col))


class Parser:
    """Recursive descent parser for VACASK format"""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.circuit = Circuit()
        self._current_subckt: Optional[Subcircuit] = None

    def current(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def peek(self, offset: int = 0) -> Token:
        pos = self.pos + offset
        return self.tokens[pos] if pos < len(self.tokens) else self.tokens[-1]

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def skip_newlines(self):
        while self.current().type == "NL":
            self.advance()

    def expect(self, type_: str) -> Token:
        tok = self.current()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_}, got {tok.type} '{tok.value}' at line {tok.line}")
        return self.advance()

    def _is_title_token(self) -> bool:
        """Check if current token can be part of a title line"""
        tok = self.current()
        # Accept NAMEs, STRINGs, and keywords (they could be part of title text)
        return tok.type in ("NAME", "STRING") or tok.type.startswith("KW_")

    def parse(self) -> Circuit:
        self.skip_newlines()
        # First non-empty line might be a title (if NAME followed by neither LPAREN nor EQ)
        if self.current().type == "NAME" and self.peek(1).type not in ("LPAREN", "EQ"):
            # Collect title until newline - accept NAMEs, STRINGs, and keywords
            title_parts = []
            while self._is_title_token() and self.current().type != "NL":
                title_parts.append(self.advance().value)
            self.circuit.title = " ".join(title_parts)
            self.skip_newlines()

        while self.current().type != "EOF":
            self.statement()
            self.skip_newlines()
        return self.circuit

    def statement(self):
        tok = self.current()

        if tok.type == "KW_LOAD":
            self.load_stmt()
        elif tok.type == "KW_INCLUDE":
            self.include_stmt()
        elif tok.type == "KW_GLOBAL":
            self.global_stmt()
        elif tok.type == "KW_GROUND":
            self.ground_stmt()
        elif tok.type == "KW_MODEL":
            self.model_stmt()
        elif tok.type == "KW_PARAMETERS":
            self.parameters_stmt()
        elif tok.type == "KW_SUBCKT":
            self.subckt_def()
        elif tok.type == "KW_CONTROL":
            self.circuit.control = self.control_block()
        elif tok.type == "KW_EMBED":
            self.embed_block()
        elif tok.type == "DIRECTIVE":
            self.directive_block()
        elif tok.type == "NAME":
            self.instance_stmt()
        elif tok.type == "NL":
            self.advance()
        else:
            raise SyntaxError(f"Unexpected token {tok.type} '{tok.value}' at line {tok.line}")

    def load_stmt(self):
        self.expect("KW_LOAD")
        filename = self.expect("STRING").value[1:-1]  # Remove quotes
        self.circuit.loads.append(filename)

    def include_stmt(self):
        self.expect("KW_INCLUDE")
        filename = self.expect("STRING").value[1:-1]
        self.circuit.includes.append(filename)

    def global_stmt(self):
        self.expect("KW_GLOBAL")
        while self.current().type == "NAME":
            self.circuit.globals.append(self.advance().value)

    def ground_stmt(self):
        self.expect("KW_GROUND")
        self.circuit.ground = self.expect("NAME").value

    def model_stmt(self):
        self.expect("KW_MODEL")
        name = self.expect("NAME").value
        module = self.expect("NAME").value
        params = {}
        # Check for parenthesized parameters (can span multiple lines)
        if self.current().type == "LPAREN":
            self.advance()  # consume (
            self.skip_newlines()
            params = self.param_list_multiline()
            self.skip_newlines()
            self.expect("RPAREN")
        elif self.current().type == "NAME" and self.peek(1).type == "EQ":
            params = self.param_list()
        self.circuit.models[name] = Model(name=name, module=module, params=params)

    def parameters_stmt(self):
        self.expect("KW_PARAMETERS")
        params = self.param_list()
        if self._current_subckt:
            self._current_subckt.params.update(params)
        else:
            self.circuit.params.update(params)

    def subckt_def(self):
        self.expect("KW_SUBCKT")
        name = self.expect("NAME").value
        self.expect("LPAREN")
        terminals = []
        while self.current().type == "NAME":
            terminals.append(self.advance().value)
        self.expect("RPAREN")

        self._current_subckt = Subcircuit(name=name, terminals=terminals)
        self.skip_newlines()

        # Parse subcircuit body
        while self.current().type not in ("KW_ENDS", "EOF"):
            if self.current().type == "KW_PARAMETERS":
                self.parameters_stmt()
            elif self.current().type == "KW_MODEL":
                self.model_stmt()  # Models can be defined inside subcircuits
            elif self.current().type == "DIRECTIVE":
                self.directive_block()  # Handle @if, @else, @endif etc.
            elif self.current().type == "NAME":
                self.instance_stmt()
            elif self.current().type == "NL":
                self.advance()
            else:
                break
            self.skip_newlines()

        self.expect("KW_ENDS")
        self.circuit.subckts[name] = self._current_subckt
        self._current_subckt = None

    def instance_stmt(self):
        name = self.expect("NAME").value
        self.expect("LPAREN")
        terminals = []
        # Terminal list can span multiple lines
        while self.current().type in ("NAME", "NL"):
            if self.current().type == "NL":
                self.advance()
            else:
                terminals.append(self.advance().value)
        self.expect("RPAREN")
        model = self.expect("NAME").value
        params = (
            self.param_list() if self.current().type == "NAME" and self.peek(1).type == "EQ" else {}
        )

        inst = Instance(name=name, terminals=terminals, model=model, params=params)
        if self._current_subckt:
            self._current_subckt.instances.append(inst)
        else:
            self.circuit.top_instances.append(inst)

    def param_list(self) -> Dict[str, str]:
        params = {}
        while self.current().type == "NAME" and self.peek(1).type == "EQ":
            name = self.advance().value
            self.expect("EQ")
            value = self.param_value()
            params[name] = value
        return params

    def param_list_multiline(self) -> Dict[str, str]:
        """Parse parameter list that can span multiple lines"""
        params = {}
        while True:
            self.skip_newlines()
            if self.current().type != "NAME":
                break
            if self.peek(1).type != "EQ":
                break
            name = self.advance().value
            self.expect("EQ")
            value = self.param_value()
            params[name] = value
        return params

    def param_value(self) -> str:
        """Parse a parameter value (may be string, name, value expression)"""
        tok = self.current()
        if tok.type == "STRING":
            return self.advance().value
        elif tok.type == "NAME":
            return self.advance().value
        elif tok.type == "VALUE":
            return self.advance().value
        else:
            raise SyntaxError(f"Expected value, got {tok.type} at line {tok.line}")

    def control_block(self) -> ControlBlock:
        """Parse control block with all VACASK commands.

        Similar to VACASK CommandInterpreter::run() (cmd.cpp:109-245).
        Parses: options, analysis, save, var, print, alter, postprocess, abort, clear
        """
        self.expect("KW_CONTROL")
        self.skip_newlines()

        control = ControlBlock()

        while self.current().type not in ("KW_ENDC", "EOF"):
            tok = self.current()

            if tok.type == "KW_OPTIONS":
                options = self._parse_options_directive()
                if control.options is None:
                    control.options = options
                else:
                    # Merge multiple options directives
                    control.options.params.update(options.params)

            elif tok.type == "KW_ANALYSIS":
                control.analyses.append(self._parse_analysis_directive())

            elif tok.type == "KW_SAVE":
                control.saves.append(self._parse_save_directive())

            elif tok.type == "KW_VAR":
                control.vars.append(self._parse_var_directive())

            elif tok.type == "KW_PRINT":
                control.prints.append(self._parse_print_directive())

            elif tok.type == "KW_POSTPROCESS":
                self._skip_postprocess()

            elif tok.type == "KW_ALTER":
                self._skip_alter()

            elif tok.type in ("KW_ABORT", "KW_CLEAR", "KW_ELABORATE"):
                self._skip_to_newline()

            elif tok.type == "NL":
                self.advance()

            else:
                # Skip unknown tokens
                self.advance()

            self.skip_newlines()

        self.expect("KW_ENDC")
        return control

    def _parse_options_directive(self) -> OptionsDirective:
        """Parse: options tran_method="trap" reltol=1e-3 abstol=1e-9"""
        self.expect("KW_OPTIONS")
        params = self.param_list()
        return OptionsDirective(params=params)

    def _parse_analysis_directive(self) -> AnalysisDirective:
        """Parse: analysis tran1 tran step=0.05n stop=1u maxstep=0.05n icmode="op" """
        self.expect("KW_ANALYSIS")
        name = self.expect("NAME").value
        analysis_type = self.expect("NAME").value
        params = self.param_list()
        return AnalysisDirective(name=name, analysis_type=analysis_type, params=params)

    def _parse_save_directive(self) -> SaveDirective:
        """Parse: save v(out) i(r1) ..."""
        self.expect("KW_SAVE")
        signals = []
        while self.current().type not in ("NL", "EOF", "KW_ENDC"):
            # Save directives can have various forms: v(node), i(device), etc.
            if self.current().type == "NAME":
                signal = self.advance().value
                # Check for function-like syntax: v(node)
                if self.current().type == "LPAREN":
                    self.advance()
                    args = []
                    while self.current().type not in ("RPAREN", "EOF"):
                        if self.current().type == "NAME":
                            args.append(self.advance().value)
                        elif self.current().type == "COMMA":
                            self.advance()
                        else:
                            break
                    if self.current().type == "RPAREN":
                        self.advance()
                    signal = f"{signal}({','.join(args)})"
                signals.append(signal)
            else:
                break
        return SaveDirective(signals=signals)

    def _parse_var_directive(self) -> VarDirective:
        """Parse: var TEMP=27 VDD=1.8"""
        self.expect("KW_VAR")
        vars_dict = self.param_list()
        return VarDirective(vars=vars_dict)

    def _parse_print_directive(self) -> PrintDirective:
        """Parse: print devices, print model "name", print stats"""
        self.expect("KW_PRINT")
        subcommand = ""
        args = []

        if self.current().type == "NAME":
            subcommand = self.advance().value

        # Collect optional arguments (strings or names)
        while self.current().type in ("NAME", "STRING"):
            val = self.advance().value
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            args.append(val)

        return PrintDirective(subcommand=subcommand, args=args)

    def _skip_postprocess(self):
        """Skip postprocess block (Python code between <<< and >>>)"""
        self.expect("KW_POSTPROCESS")
        # Skip until newline or end of control
        while self.current().type not in ("NL", "EOF", "KW_ENDC"):
            # Check for heredoc-style blocks
            if self.current().type == "NAME" and self.current().value == "<<<":
                self.advance()
                # Skip until >>>
                while self.current().type != "EOF":
                    if self.current().type == "NAME" and self.current().value == ">>>":
                        self.advance()
                        break
                    self.advance()
            else:
                self.advance()

    def _skip_alter(self):
        """Skip alter command"""
        self.expect("KW_ALTER")
        self._skip_to_newline()

    def _skip_to_newline(self):
        """Skip tokens until newline or end of control"""
        while self.current().type not in ("NL", "EOF", "KW_ENDC"):
            self.advance()

    def embed_block(self):
        """Skip embed block"""
        self.expect("KW_EMBED")
        # Skip until we see >>>FILE or EOF
        while self.current().type != "EOF":
            if ">>>FILE" in self.current().value:
                self.advance()
                break
            self.advance()

    def directive_block(self):
        """Skip @if/@elseif/@else/@endif/@end conditional block"""
        tok = self.current()
        if tok.value.lower() in ("@endif", "@end"):
            # Standalone @endif/@end - just consume it
            self.advance()
            return
        if tok.value.lower() in ("@elseif", "@else"):
            # Part of an @if block - consume and skip to newline
            self.advance()
            while self.current().type not in ("NL", "EOF"):
                self.advance()
            return

        # @if - skip entire conditional block until matching @endif/@end
        self.advance()  # consume @if
        # Skip condition to end of line
        while self.current().type not in ("NL", "EOF"):
            self.advance()

        # Skip content until matching @endif/@end (handling nested @if)
        depth = 1
        while depth > 0 and self.current().type != "EOF":
            self.skip_newlines()
            if self.current().type == "DIRECTIVE":
                directive = self.current().value.lower()
                if directive == "@if":
                    depth += 1
                    self.advance()
                    # Skip rest of line (condition)
                    while self.current().type not in ("NL", "EOF"):
                        self.advance()
                elif directive in ("@endif", "@end"):
                    depth -= 1
                    self.advance()
                elif directive in ("@elseif", "@else"):
                    self.advance()
                    # Skip rest of line
                    while self.current().type not in ("NL", "EOF"):
                        self.advance()
                else:
                    self.advance()
            else:
                # Skip any statement inside the block
                self.advance()


class VACASKParser:
    """Parser for VACASK netlist format"""

    def parse(self, text: str, base_path: Optional[Path] = None) -> Circuit:
        """Parse VACASK netlist text"""
        lexer = Lexer(text)
        parser = Parser(lexer.tokens)
        return parser.parse()

    def parse_file(self, filename: Union[str, Path]) -> Circuit:
        """Parse VACASK netlist from file, handling includes"""
        path = Path(filename)
        text = path.read_text()
        circuit = self.parse(text, path.parent)

        # Process includes
        for include in circuit.includes[:]:
            include_path = path.parent / include
            if include_path.exists():
                included = self.parse_file(include_path)
                circuit.models.update(included.models)
                circuit.subckts.update(included.subckts)
                circuit.params.update(included.params)
                if included.globals:
                    circuit.globals.extend(g for g in included.globals if g not in circuit.globals)
                if included.ground and not circuit.ground:
                    circuit.ground = included.ground

        return circuit


def parse_netlist(source: Union[str, Path]) -> Circuit:
    """Convenience function to parse a VACASK netlist"""
    parser = VACASKParser()
    if isinstance(source, Path):
        return parser.parse_file(source)
    elif isinstance(source, str) and "\n" not in source and Path(source).exists():
        return parser.parse_file(Path(source))
    else:
        return parser.parse(source)
