import re

# End of line comment
pat_eolcomment = re.compile(r"(\$|;|//)")

# leading whitespace
pat_leadspace = re.compile(r"^\s+")

# Case insensitive dot directive patterns
pat_cidotmodel = re.compile(r"\.model", re.IGNORECASE)
pat_cidotinclude = re.compile(r"\.include", re.IGNORECASE)
pat_cidotlib = re.compile(r"\.lib", re.IGNORECASE)
pat_cidotendl = re.compile(r"\.endl", re.IGNORECASE)
pat_cidotcontrol = re.compile(r"\.control", re.IGNORECASE)
pat_cidotendc = re.compile(r"\.endc", re.IGNORECASE)
pat_cidotsubckt = re.compile(r"\.subckt", re.IGNORECASE)
pat_cidotends = re.compile(r"\.ends", re.IGNORECASE)
pat_cidotparams = re.compile(r"\.params", re.IGNORECASE)
pat_cidotparam = re.compile(r"\.param", re.IGNORECASE)
pat_cidotif = re.compile(r"\.if", re.IGNORECASE)
pat_cidotelseif = re.compile(r"\.elseif", re.IGNORECASE)
pat_cidotelse = re.compile(r"\.else", re.IGNORECASE)
pat_cidotendif = re.compile(r"\.endif", re.IGNORECASE)
pat_cidotend = re.compile(r"\.end", re.IGNORECASE)

# pre_osdi pattern
pat_preosdi = re.compile(r"pre_osdi")

# .end with leading and trailing whitespace, whole line
pat_cidotend_line = re.compile(r"\s*\.end\s*$", re.IGNORECASE)

# Something in single quotes
pat_singlequotes = re.compile(r"'(.*?)'")

# Sequence of 1 or more spaces
pat_spaceseq = re.compile(r" +")

# whitespace before and after =
pat_spaceequal = re.compile(r"\s*=\s*")

# Pair of curly braces with anything inside
pat_cbracepair = re.compile(r"\{(.*?)\}")
