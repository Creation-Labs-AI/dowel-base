"""Contains the output classes for the logger.

Each class is sent logger data and handles it itself.
"""
import abc
import datetime
import os
import sys
from typing import Optional, TypeVar, Union

import dateutil.tz

from dowel.log_output import LogOutput
from dowel.tabular_input import TabularInput

def data_as_string(data: Union[str, TabularInput], prefix: str, with_timestamp: bool):
    if isinstance(data, str):
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
            return f"{timestamp} | {prefix}{data}"
        else:
            return f"{prefix}{data}"
    elif isinstance(data, TabularInput):
        data_str = str(data)
        data.mark_str()

        return data_str
    else:
        raise ValueError('Unacceptable type.')

class StdOutput(LogOutput[Union[str, TabularInput]]):
    """Standard console output for the logger.

    :param with_timestamp: Whether to log a timestamp before non-tabular data.
    """

    def __init__(self, with_timestamp: bool = True):
        super().__init__()

        self._with_timestamp = with_timestamp

    @property
    def types_accepted(self):
        """Accept str and TabularInput objects."""
        return (str, TabularInput)

    def record(self, data: Union[str, TabularInput], prefix: str = ''):
        """Log data to console."""
        print(data_as_string(data, prefix, self._with_timestamp))

    def dump(self, step: Optional[int] = None):
        """Flush data to standard output stream."""
        sys.stdout.flush()

FileOutputRecordType = TypeVar("FileOutputRecordType")

class FileOutput(LogOutput[FileOutputRecordType], metaclass=abc.ABCMeta):
    """File output abstract class for logger.

    :param file_name: The file this output should log to.
    :param mode: File open mode ('a', 'w', etc).
    """

    def __init__(self, file_name: str, mode: str = "w"):
        super().__init__()

        if self._fs.protocol == "file":
            dir_path = os.path.dirname(file_name)
            self._fs.makedirs(dir_path, exist_ok=True)

        self.mode = mode
        self.file_name = file_name

        self._log_file = self.open_log_file()

    def open_log_file(self):
        return self._fs.open(self.file_name, self.mode)

    def close(self):
        self._log_file.close()

    def dump(self, step: Optional[int] = None):
        """Flush data to log file."""
        self._log_file.flush()

class TextOutput(FileOutput[Union[str, TabularInput]]):
    """Text file output for logger.

    :param file_name: The file this output should log to.
    :param with_timestamp: Whether to log a timestamp before the data.
    """

    def __init__(self, file_name: str, with_timestamp: bool = True):
        super().__init__(file_name, 'a')

        self._with_timestamp = with_timestamp
        self._delimiter = ' | '

    @property
    def types_accepted(self):
        """Accept str objects only."""
        return (str, TabularInput)

    def record(self, data: Union[str, TabularInput], prefix=''):
        """Log data to text file."""
        self._log_file.write(f"{data_as_string(data, prefix, self._with_timestamp)}\n")
