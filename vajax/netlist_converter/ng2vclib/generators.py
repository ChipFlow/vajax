from .patterns import *


# A generator that traverses a deck
def traverse(deck, depth=None, input_history=[], parent_line=None, inside_control=False):
    """
    A generator that traverses a *deck*.

    *depth* is the lowest level to which we traverse the deck.
    0 dumps only the toplevel file. If *depth* is None the deck
    is traveresed all the way to the bottom.

    *input_history* is the history up to the *deck*.

    *parent_line* is the line number on which this *deck* was included.

    *inside_control* is a flag specifying whether the deck is inside
    a .control block.

    It yields
    * a list of history items which are tuples holding
      * line number in parent file where this file was included,
        None for the toplevel file.
      * filename
      * section
    * a line which is a tuple holding
      * line number
      * leading whitespace
      * core line
      * eol comment .. a string or a tuple if the line corresponds
        to a .include or .lib directive
    * depth at which the line is located
    * a flag indicating this line is a part of control clock
    """
    filename, section, lines = deck
    history = input_history + [(parent_line, filename, section)]

    at_depth = len(history) - 1

    for line in lines:
        lnum, lws, l, eolc, annot = line
        # Check for control
        ll = l.lower()
        if pat_cidotcontrol.match(ll):
            inside_control = True
        elif pat_cidotendc.match(ll):
            inside_control = False

        # Is it a normal line
        if not isinstance(eolc, tuple):
            # Yes
            yield (history, line, depth, inside_control)
        else:
            # No
            # First yield inclusion line
            yield (history, line, at_depth, inside_control)

            # Check if we are not at the bottom
            if depth is None or at_depth < depth:
                # Do we have a subdeck
                _, _, sublines = eolc
                if sublines is not None:
                    # Yield subdeck
                    for subh, subl, subd, inside_control in traverse(
                        eolc, depth, history, lnum, inside_control=inside_control
                    ):
                        yield (subh, subl, subd, inside_control)


def format_history(history, lineno):
    """
    Formats *history* for an error on line number *lineno*.
    """
    txt = "On line " + str(lineno + 1) + " of "
    for h in reversed(history):
        pline, fname, sec = h
        txt += "'" + fname + "'"
        if pline is not None:
            if sec is not None:
                txt += "included as section" + sec + " on line " + str(pline + 1) + " of "
            else:
                txt += "included on line " + str(pline + 1) + " of "

    return txt
