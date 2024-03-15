from typing import List, Optional, Tuple, Dict
from models.DataNecessityType import *


class NotEqualSetDataTypesError(Exception):
    """ Set data defined with different types """
    pass


class NotEqualSetDataLenError(Exception):
    """ Set data defined with different length """
    pass


class DataSetValueIsNoneTypeError(Exception):
    """ datasets value is none """
    pass


class DataSetValuesIsNoneTypeError(Exception):
    """ datasets values is none """
    pass


class DataSetObject:
    def __init__(
            self,
            input_data: Optional[Dict[str, List[float]]],
            output_data: Optional[List[float]],
            data_necessity_type: DataNecessityType,
            ignore_errors: Tuple = ()
    ):
        self.error_occurred_on_init: Optional[Exception] = None
        try:
            if not (input_data is not None and output_data is not None):
                if input_data is None and output_data is None:
                    raise DataSetValuesIsNoneTypeError("Input and output data is None.")
                else:
                    raise DataSetValueIsNoneTypeError("Both input and output data should be specified together.")
            if len(output_data) != len(list(input_data.values())[0]):
                raise NotEqualSetDataLenError("Input data should contain same number of values as output data.")
            if not isinstance(output_data, type(list(input_data.values())[0])):  # type(input_data) != type(output_data[0]):
                raise NotEqualSetDataTypesError("Input data should contain same number of values as output data.")
        except ignore_errors as e:
            self.error_occurred_on_init = e

        self.output_data = output_data
        self.data_necessity_type = data_necessity_type
        self.ignored_errors = ignore_errors

        if not self.error_occurred_on_init:
            self.input_data = list(input_data.values())

            self.shape = (len(input_data), 1)
            self.data_len = len(self.input_data[0])
            self.keys = list(input_data.keys())
