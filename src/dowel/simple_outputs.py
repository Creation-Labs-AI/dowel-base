"""Contains the output classes for the logger.

Each class is sent logger data and handles it itself.
"""
import abc
import datetime
import os
import sys
import dateutil.tz

from dowel import LogOutput
from dowel.tabular_input import TabularInput

class StdOutput(LogOutput):
    """Standard console output for the logger.

    :param with_timestamp: Whether to log a timestamp before non-tabular data.
    """

    def __init__(self, with_timestamp=True):
        super().__init__()
        self._with_timestamp = with_timestamp

    @property
    def types_accepted(self):
        """Accept str and TabularInput objects."""
        return (str, TabularInput)

    def record(self, data, prefix=''):
        """Log data to console."""
        if isinstance(data, str):
            out = prefix + data
            if self._with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = '%s | %s' % (timestamp, out)
        elif isinstance(data, TabularInput):
            out = str(data)
            data.mark_str()
        else:
            raise ValueError('Unacceptable type')

        print(out)

    def dump(self, step=None):
        """Flush data to standard output stream."""
        sys.stdout.flush()


class FileOutput(LogOutput, metaclass=abc.ABCMeta):
    """File output abstract class for logger.

    :param file_name: The file this output should log to.
    :param mode: File open mode ('a', 'w', etc).
    """

    def __init__(self, file_name, mode='w'):
        super().__init__()
        if self._fs.protocol == "file":
            dir_path = os.path.dirname(file_name)
            self._fs.makedirs(dir_path, exist_ok=True)

        self.mode = mode
        self.file_name = file_name

class TextOutput(FileOutput):
    """Text file output for logger.

    :param file_name: The file this output should log to.
    :param with_timestamp: Whether to log a timestamp before the data.
    """

    def __init__(self, file_name, with_timestamp=True):
        super().__init__(file_name, 'a')
        self._with_timestamp = with_timestamp
        self._delimiter = ' | '

    @property
    def types_accepted(self):
        """Accept str objects only."""
        return (str, TabularInput)

    def record(self, data, prefix=''):
        """Log data to text file."""
        if isinstance(data, str):
            out = prefix + data
            if self._with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = '%s | %s' % (timestamp, out)
        elif isinstance(data, TabularInput):
            out = str(data)
            data.mark_str()
        else:
            raise ValueError('Unacceptable type.')
        
        with self._fs.open(self.file_name, self.mode) as fo: 
            fo.write(out + '\n')

