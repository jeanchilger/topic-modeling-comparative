COLOR_CODES = {
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}

END = "\033[0m"
BOLD = "1"
UNDERLINE = "4"

def format_stdout_string(
        text, color="white", bold=False, 
        underline=False, bright=False):
    """Formats a string to stdout.

    Applies the specified format configuration to the given
    string and returns.

    Args:
        color (str, optional): [description]. Defaults to "white".
        bold (bool, optional): [description]. Defaults to False.
        underline (bool, optional): [description]. Defaults to False.
    """

    _bright = 30 if bright else 0

    start = "\033[{}".format(COLOR_CODES[color] + _bright)

    if bold:
        start += ";{}".format(BOLD)

    if underline:
        start += ";{}".format(UNDERLINE)
    
    start += "m"

    out_str = start + text + END

    return out_str

def print_color(
        text, color="white", bold=False, 
        underline=False, bright=False):
    """Prints a colored string

    Formats the given string and prints it to the stdout.

    Args:
        color (str, optional): [description]. Defaults to "white".
        bold (bool, optional): [description]. Defaults to False.
        underline (bool, optional): [description]. Defaults to False.
    """

    print(format_stdout_string(
            text, color=color, bold=bold, 
            underline=underline, bright=bright))

    