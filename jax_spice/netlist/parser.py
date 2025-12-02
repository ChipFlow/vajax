"""VACASK netlist parser - simple recursive descent

Parses VACASK format netlists for circuit simulation.
Adapted from VACASK dflparser grammar.
"""

import re
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

from jax_spice.netlist.circuit import Circuit, Subcircuit, Instance, Model


@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int


class Lexer:
    """Simple lexer for VACASK netlist format"""

    KEYWORDS = {'load', 'include', 'global', 'ground', 'model', 'parameters',
                'subckt', 'ends', 'control', 'endc', 'embed'}

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
            if self.text[self.pos] in ' \t':
                self.pos += 1
                self.col += 1
                continue

            # Newline
            if self.text[self.pos] == '\n':
                self.tokens.append(Token('NL', '\n', self.line, self.col))
                self.pos += 1
                self.line += 1
                self.col = 1
                continue

            if self.text[self.pos] == '\r':
                self.pos += 1
                continue

            # Comment
            if self.pos + 1 < len(self.text) and self.text[self.pos:self.pos+2] == '//':
                while self.pos < len(self.text) and self.text[self.pos] != '\n':
                    self.pos += 1
                continue

            # String
            if self.text[self.pos] == '"':
                start = self.pos
                self.pos += 1
                while self.pos < len(self.text) and self.text[self.pos] != '"':
                    self.pos += 1
                self.pos += 1  # closing quote
                self.tokens.append(Token('STRING', self.text[start:self.pos], self.line, self.col))
                self.col += self.pos - start
                continue

            # Special chars
            if self.text[self.pos] == '(':
                self.tokens.append(Token('LPAREN', '(', self.line, self.col))
                self.pos += 1
                self.col += 1
                continue

            if self.text[self.pos] == ')':
                self.tokens.append(Token('RPAREN', ')', self.line, self.col))
                self.pos += 1
                self.col += 1
                continue

            if self.text[self.pos] == '=':
                self.tokens.append(Token('EQ', '=', self.line, self.col))
                self.pos += 1
                self.col += 1
                # After =, collect the value (may contain expressions with parens)
                # Skip whitespace first
                while self.pos < len(self.text) and self.text[self.pos] in ' \t':
                    self.pos += 1
                    self.col += 1
                if self.pos < len(self.text) and self.text[self.pos] == '"':
                    # String value - will be handled in next iteration
                    continue
                # Collect expression value, handling balanced parentheses
                start = self.pos
                paren_depth = 0
                while self.pos < len(self.text):
                    ch = self.text[self.pos]
                    if ch == '(':
                        paren_depth += 1
                    elif ch == ')':
                        if paren_depth == 0:
                            break  # Unmatched ) - end of value
                        paren_depth -= 1
                    elif ch in ' \t\n\r' and paren_depth == 0:
                        break  # Whitespace outside parens - end of value
                    self.pos += 1
                if self.pos > start:
                    val = self.text[start:self.pos]
                    self.tokens.append(Token('VALUE', val, self.line, self.col))
                    self.col += self.pos - start
                continue

            # Identifier or keyword or number
            if self.text[self.pos].isalnum() or self.text[self.pos] == '_':
                start = self.pos
                while self.pos < len(self.text) and (
                    self.text[self.pos].isalnum() or self.text[self.pos] in '_.'
                ):
                    self.pos += 1
                word = self.text[start:self.pos]
                if word.lower() in self.KEYWORDS:
                    self.tokens.append(Token('KW_' + word.upper(), word, self.line, self.col))
                else:
                    self.tokens.append(Token('NAME', word, self.line, self.col))
                self.col += self.pos - start
                continue

            # Value (for param=value, can include special chars)
            # Skip for now, handle in parser
            self.pos += 1
            self.col += 1

        self.tokens.append(Token('EOF', '', self.line, self.col))


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
        while self.current().type == 'NL':
            self.advance()

    def expect(self, type_: str) -> Token:
        tok = self.current()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_}, got {tok.type} '{tok.value}' at line {tok.line}")
        return self.advance()

    def parse(self) -> Circuit:
        self.skip_newlines()
        # First non-empty line might be a title (if it's just NAME tokens without keywords)
        if self.current().type == 'NAME' and self.peek(1).type not in ('LPAREN', 'EQ'):
            # Collect title until newline
            title_parts = []
            while self.current().type in ('NAME', 'STRING') and self.current().type != 'NL':
                title_parts.append(self.advance().value)
            self.circuit.title = ' '.join(title_parts)
            self.skip_newlines()

        while self.current().type != 'EOF':
            self.statement()
            self.skip_newlines()
        return self.circuit

    def statement(self):
        tok = self.current()

        if tok.type == 'KW_LOAD':
            self.load_stmt()
        elif tok.type == 'KW_INCLUDE':
            self.include_stmt()
        elif tok.type == 'KW_GLOBAL':
            self.global_stmt()
        elif tok.type == 'KW_GROUND':
            self.ground_stmt()
        elif tok.type == 'KW_MODEL':
            self.model_stmt()
        elif tok.type == 'KW_PARAMETERS':
            self.parameters_stmt()
        elif tok.type == 'KW_SUBCKT':
            self.subckt_def()
        elif tok.type == 'KW_CONTROL':
            self.control_block()
        elif tok.type == 'KW_EMBED':
            self.embed_block()
        elif tok.type == 'NAME':
            self.instance_stmt()
        elif tok.type == 'NL':
            self.advance()
        else:
            raise SyntaxError(f"Unexpected token {tok.type} '{tok.value}' at line {tok.line}")

    def load_stmt(self):
        self.expect('KW_LOAD')
        filename = self.expect('STRING').value[1:-1]  # Remove quotes
        self.circuit.loads.append(filename)

    def include_stmt(self):
        self.expect('KW_INCLUDE')
        filename = self.expect('STRING').value[1:-1]
        self.circuit.includes.append(filename)

    def global_stmt(self):
        self.expect('KW_GLOBAL')
        while self.current().type == 'NAME':
            self.circuit.globals.append(self.advance().value)

    def ground_stmt(self):
        self.expect('KW_GROUND')
        self.circuit.ground = self.expect('NAME').value

    def model_stmt(self):
        self.expect('KW_MODEL')
        name = self.expect('NAME').value
        module = self.expect('NAME').value
        params = {}
        # Check for parenthesized parameters (can span multiple lines)
        if self.current().type == 'LPAREN':
            self.advance()  # consume (
            self.skip_newlines()
            params = self.param_list_multiline()
            self.skip_newlines()
            self.expect('RPAREN')
        elif self.current().type == 'NAME' and self.peek(1).type == 'EQ':
            params = self.param_list()
        self.circuit.models[name] = Model(name=name, module=module, params=params)

    def parameters_stmt(self):
        self.expect('KW_PARAMETERS')
        params = self.param_list()
        if self._current_subckt:
            self._current_subckt.params.update(params)
        else:
            self.circuit.params.update(params)

    def subckt_def(self):
        self.expect('KW_SUBCKT')
        name = self.expect('NAME').value
        self.expect('LPAREN')
        terminals = []
        while self.current().type == 'NAME':
            terminals.append(self.advance().value)
        self.expect('RPAREN')

        self._current_subckt = Subcircuit(name=name, terminals=terminals)
        self.skip_newlines()

        # Parse subcircuit body
        while self.current().type not in ('KW_ENDS', 'EOF'):
            if self.current().type == 'KW_PARAMETERS':
                self.parameters_stmt()
            elif self.current().type == 'NAME':
                self.instance_stmt()
            elif self.current().type == 'NL':
                self.advance()
            else:
                break
            self.skip_newlines()

        self.expect('KW_ENDS')
        self.circuit.subckts[name] = self._current_subckt
        self._current_subckt = None

    def instance_stmt(self):
        name = self.expect('NAME').value
        self.expect('LPAREN')
        terminals = []
        # Terminal list can span multiple lines
        while self.current().type in ('NAME', 'NL'):
            if self.current().type == 'NL':
                self.advance()
            else:
                terminals.append(self.advance().value)
        self.expect('RPAREN')
        model = self.expect('NAME').value
        params = self.param_list() if self.current().type == 'NAME' and self.peek(1).type == 'EQ' else {}

        inst = Instance(name=name, terminals=terminals, model=model, params=params)
        if self._current_subckt:
            self._current_subckt.instances.append(inst)
        else:
            self.circuit.top_instances.append(inst)

    def param_list(self) -> Dict[str, str]:
        params = {}
        while self.current().type == 'NAME' and self.peek(1).type == 'EQ':
            name = self.advance().value
            self.expect('EQ')
            value = self.param_value()
            params[name] = value
        return params

    def param_list_multiline(self) -> Dict[str, str]:
        """Parse parameter list that can span multiple lines"""
        params = {}
        while True:
            self.skip_newlines()
            if self.current().type != 'NAME':
                break
            if self.peek(1).type != 'EQ':
                break
            name = self.advance().value
            self.expect('EQ')
            value = self.param_value()
            params[name] = value
        return params

    def param_value(self) -> str:
        """Parse a parameter value (may be string, name, value expression)"""
        tok = self.current()
        if tok.type == 'STRING':
            return self.advance().value
        elif tok.type == 'NAME':
            return self.advance().value
        elif tok.type == 'VALUE':
            return self.advance().value
        else:
            raise SyntaxError(f"Expected value, got {tok.type} at line {tok.line}")

    def control_block(self):
        """Skip control block"""
        self.expect('KW_CONTROL')
        depth = 1
        while depth > 0 and self.current().type != 'EOF':
            if self.current().type == 'KW_CONTROL':
                depth += 1
            elif self.current().type == 'KW_ENDC':
                depth -= 1
            self.advance()

    def embed_block(self):
        """Skip embed block"""
        self.expect('KW_EMBED')
        # Skip until we see >>>FILE or EOF
        while self.current().type != 'EOF':
            if '>>>FILE' in self.current().value:
                self.advance()
                break
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
    elif isinstance(source, str) and '\n' not in source and Path(source).exists():
        return parser.parse_file(Path(source))
    else:
        return parser.parse(source)
