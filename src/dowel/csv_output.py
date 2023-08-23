"""A `dowel.logger.LogOutput` for CSV files."""
import csv
import os

from dowel.simple_outputs import FileOutput
from dowel.tabular_input import TabularInput

class CsvOutput(FileOutput[TabularInput]):
    """CSV file output for logger.

    :param file_name: The file this output should log to.
    """

    def __init__(self, file_name: str):
        super().__init__(file_name)

        self._writer = None
        self._fieldnames = None
        self._filename = file_name

    @property
    def types_accepted(self):
        """Accept TabularInput objects only."""
        return (TabularInput,)

    def record(self, data: TabularInput, prefix: str = ''):
        """Log tabular data to CSV."""
        if isinstance(data, TabularInput):
            to_csv = data.as_primitive_dict

            if not self._writer:
                self._fieldnames = set(to_csv.keys())
                self._writer = csv.DictWriter(self._log_file,
                                              fieldnames=self._fieldnames,
                                              restval='',
                                              extrasaction='raise')
                self._writer.writeheader()

            if to_csv.keys() != self._fieldnames:
                # Close existing log file
                super().close()

                # Move log file to temp file
                temp_file_name = '{}.tmp'.format(self._filename)
                os.replace(self._filename, temp_file_name)

                # Add new keys to fieldnames
                self._fieldnames = (set(self._fieldnames) | set(to_csv.keys()))

                # Open a new copy of the log file
                self._log_file = self.open_log_file()
                self._writer = csv.DictWriter(self._log_file,
                                              fieldnames=self._fieldnames,
                                              restval='',
                                              extrasaction='raise')

                # Transfer data from temp file
                with open(temp_file_name, 'r') as temp_file:
                    self._writer.writeheader()
                    for row in csv.DictReader(temp_file):
                        self._writer.writerow(row)

            self._writer.writerow(to_csv)

            for k in to_csv.keys():
                data.mark(k)
        else:
            raise ValueError('Unacceptable type.')