"""A `dowel.logger.LogOutput` for CSV files."""
import pandas as pd

from dowel import TabularInput
from dowel.simple_outputs import FileOutput

class CsvOutput(FileOutput):
    """CSV file output for logger.

    # TODO: Add buffering at some point to reduce S3 traffic

    :param file_name: The file this output should log to.
    """

    def __init__(self, file_name):
        super().__init__(file_name)

    @property
    def types_accepted(self):
        """Accept TabularInput objects only."""
        return (TabularInput, )
    
    def _save_as_csv(self, data: pd.DataFrame) -> None:
        with self._fs.open(self.file_name, self.mode) as fo:
            data.to_csv(fo, index=False)

    def record(self, data, prefix=''):
        """Log tabular data to CSV."""
        if not isinstance(data, TabularInput):
            raise ValueError('Unacceptable type.')
        
        to_csv = {k: [v] for k, v in data.as_primitive_dict.items()}
        new_df = pd.DataFrame.from_dict(to_csv)

        # if file does not exist, create it and save passed data
        if not self._fs.exists(self.file_name):
            self._save_as_csv(new_df)
            return 
        
        # if file exists, read it and check if all columns are present
        with self._fs.open(self.file_name, 'r') as fi:
            curr_df = pd.read_csv(fi)
        
        if to_csv.keys() != set(curr_df.columns):
            # add new columns to dataframe
            for key in to_csv.keys():
                if key not in curr_df.columns:
                    curr_df[key] = None

        # add new row to dataframe and save to csv
        df = pd.concat([curr_df, new_df], ignore_index=True)
        self._save_as_csv(df)

        for k in to_csv.keys():
            data.mark(k)