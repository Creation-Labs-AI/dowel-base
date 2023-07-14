import fsspec

_FILESYSTEM = fsspec.filesystem(protocol='file')

def set_filesystem(fs: fsspec.AbstractFileSystem):
    """Set the filesystem to use for logging.

    :param fs: The filesystem to use for logging.
    """
    global _FILESYSTEM
    _FILESYSTEM = fs

def get_filesystem() -> fsspec.AbstractFileSystem:
    """Get the filesystem used for logging.

    :return: The filesystem used for logging.
    """
    return _FILESYSTEM
