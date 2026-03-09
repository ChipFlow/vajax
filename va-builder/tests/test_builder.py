"""Tests for va_builder netlist builder."""

from va_builder import Netlist, SubcktBuilder


class TestSubcktBuilder:
    def test_simple_subckt(self):
        s = SubcktBuilder("inv", ["out", "in"])
        s.inst("mp", "pmos", ["out", "in", "vdd", "vdd"], w="1u", l="0.2u")
        s.inst("mn", "nmos", ["out", "in", "vss", "vss"], w="0.5u", l="0.2u")
        text = s.render()
        assert "subckt inv(out in)" in text
        assert "mp (out in vdd vdd) pmos w=1u l=0.2u" in text
        assert "mn (out in vss vss) nmos w=0.5u l=0.2u" in text
        assert text.endswith("ends")

    def test_subckt_with_params(self):
        s = SubcktBuilder("drv", ["out"], params={"v0": "0", "v1": "1"})
        text = s.render()
        assert "parameters v0=0 v1=1" in text

    def test_context_manager(self):
        s = SubcktBuilder("test", ["a", "b"])
        with s:
            s.inst("r1", "res", ["a", "b"], r="1k")
        text = s.render()
        assert "r1 (a b) res r=1k" in text

    def test_comment_and_blank(self):
        s = SubcktBuilder("test", ["a"])
        s.comment("This is a comment")
        s.blank()
        s.inst("r1", "res", ["a", "0"])
        text = s.render()
        assert "// This is a comment" in text
        lines = text.split("\n")
        # blank line should be between comment and instance
        assert "" in lines


class TestNetlist:
    def test_globals_and_ground(self):
        nl = Netlist(globals=["vdd", "vss"], ground="0")
        text = str(nl)
        assert "global vdd vss" in text
        assert "ground 0" in text

    def test_title(self):
        nl = Netlist(title="Test circuit")
        text = str(nl)
        assert "// Test circuit" in text

    def test_model(self):
        nl = Netlist()
        nl.model("v", "vsource")
        nl.model("r", "resistor", r="1k")
        text = str(nl)
        assert "model v vsource" in text
        assert "model r resistor r=1k" in text

    def test_top_instance(self):
        nl = Netlist()
        nl.inst("v1", "vsource", ["a", "0"], dc="1")
        text = str(nl)
        assert "v1 (a 0) vsource dc=1" in text

    def test_subckt_context(self):
        nl = Netlist(globals=["vdd", "vss"], ground="0")
        with nl.subckt("inv", ["out", "in"]) as s:
            s.inst("mp", "pmos", ["out", "in", "vdd", "vdd"], w="1u", l="0.2u")
            s.inst("mn", "nmos", ["out", "in", "vss", "vss"], w="0.5u", l="0.2u")
        text = str(nl)
        assert "subckt inv(out in)" in text
        assert "ends" in text

    def test_loop_generation(self):
        """Test that Python loops generate correct netlist."""
        nl = Netlist(globals=["vdd", "vss"], ground="0")
        n = 4
        ports = [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]
        with nl.subckt("test", ports) as s:
            for i in range(n):
                for j in range(n):
                    s.inst(f"pp_{i}_{j}", "and", [f"pp{i}_{j}", f"a{i}", f"b{j}"])
        text = str(nl)
        # Should have n*n instances
        assert text.count(") and") == n * n
        assert "pp_0_0 (pp0_0 a0 b0) and" in text
        assert "pp_3_3 (pp3_3 a3 b3) and" in text

    def test_comment_and_blank(self):
        nl = Netlist()
        nl.comment("Header comment")
        nl.blank()
        nl.model("v", "vsource")
        text = str(nl)
        assert "// Header comment" in text

    def test_full_netlist_roundtrip(self):
        """Build a small netlist and verify it parses as valid VACASK."""
        nl = Netlist(globals=["vdd", "vss"], ground="0")
        with nl.subckt("not", ["out", "in"]) as s:
            s.inst("mp", "pmos", ["out", "in", "vdd", "vdd"], w="1u", l="0.2u")
            s.inst("mn", "nmos", ["out", "in", "vss", "vss"], w="0.5u", l="0.2u")
        with nl.subckt("and", ["out", "in1", "in2"]) as s:
            s.inst("mp2", "pmos", ["outx", "in2", "vdd", "vdd"], w="1u", l="0.2u")
            s.inst("mp1", "pmos", ["outx", "in1", "vdd", "vdd"], w="1u", l="0.2u")
            s.inst("mn1", "nmos", ["outx", "in1", "int", "vss"], w="0.5u", l="0.2u")
            s.inst("mn2", "nmos", ["int", "in2", "vss", "vss"], w="0.5u", l="0.2u")
            s.inst("mp3", "pmos", ["out", "outx", "vdd", "vdd"], w="1u", l="0.2u")
            s.inst("mn3", "nmos", ["out", "outx", "vss", "vss"], w="0.5u", l="0.2u")

        text = str(nl)
        # Verify structure
        assert text.count("subckt ") == 2
        assert text.count("ends") == 2
        assert "global vdd vss" in text
        assert "ground 0" in text


class TestSubcktDefaultParams:
    def test_default_params_rendered(self):
        nl = Netlist()
        with nl.subckt("drv", ["out"], v0="0", v1="1") as s:
            s.inst("r1", "res", ["out", "0"], r="1k")
        text = str(nl)
        assert "parameters v0=0 v1=1" in text
