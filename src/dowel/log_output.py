import abc
from typing import Generic, Optional, TypeVar

from dowel.filesystem import get_filesystem

RecordType = TypeVar("RecordType")

class LogOutput(abc.ABC, Generic[RecordType]):
    """Abstract class for Logger Outputs."""

    def __init__(self):
        self._fs = get_filesystem() # initialize filesystem
        self.protocol = self._fs.protocol if isinstance(self._fs.protocol, str) else self._fs.protocol[0]

        assert self.protocol in ['file', 's3'], "Only file and s3 protocols are supported"

    @abc.abstractproperty
    @property
    def types_accepted(self):
        """Pass these types to this logger output.

        The types in this tuple will be accepted by this output.

        :return: A tuple containing all valid input types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def record(self, data: RecordType, prefix: str = ''):
        """Pass logger data to this output.

        :param data: The data to be logged by the output.
        :param prefix: A prefix placed before a log entry in text outputs.
        """
        pass

    def dump(self, step: Optional[int] = None):
        """Dump the contents of this output.

        :param step: The current run step.
        """
        pass

    def close(self):
        """Close any files used by the output."""
        pass

    def __del__(self):
        """Clean up object upon deletion."""
        self.close()