"""Utilities for console outputs."""
import errno
import os
color2num = dict(gray=30,
                 red=31,
                 green=32,
                 yellow=33,
                 blue=34,
                 magenta=35,
                 cyan=36,
                 white=37,
                 crimson=38)


def colorize(string, color, bold=False, highlight=False):
    """Colorize the string for console output."""
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def get_relative_path(path: str):
    """Get the relative path of a file path given the current working directory.
    The path is given in the form '{protocol}://{absolute_prefix}/{path}'. The
    protocol can be 'file', 's3', etc.
    Args:
        path: The absolute path of the file.
        
    Returns:
        The relative path of the file.
    """
    if path.startswith("file://"):
        path = path[len("file://"):]
        return os.path.relpath(path, os.getcwd())
    else:
        path = path[len("s3://"):]
        return '/'.join(path.split('/')[1:])