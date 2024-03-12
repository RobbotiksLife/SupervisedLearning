from typing import List, Any, Optional, Tuple
from DataNecessityType import *


class NotEqualSetDataTypesError(Exception):
    """ Set data defined with different types """
    pass

class NotEqualSetDataLenError(Exception):
    """ Set data defined with different length """
    pass

class DataSetValueIsNoneTypeError(Exception):
    """ dataset value is none """
    pass

class DataSetValuesIsNoneTypeError(Exception):
    """ dataset values is none """
    pass


class DataSetObject:
    def __init__(
            self,
            input_data: Optional[List[Any]],
            output_data: Optional[List[Any]],
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
            if len(input_data) != len(output_data):
                raise NotEqualSetDataLenError("Input data should contain same number of values as output data.")
            if type(input_data) != type(output_data):
                raise NotEqualSetDataTypesError("Input data should contain same number of values as output data.")
        except ignore_errors as e:
            self.error_occurred_on_init = e

        self.input_data = input_data
        self.output_data = output_data
        self.data_necessity_type = data_necessity_type
        self.ignored_errors = ignore_errors

        if not self.error_occurred_on_init:
            self.data_len = len(input_data)
