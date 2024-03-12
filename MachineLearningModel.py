from typing import Dict
from DataSetObject import *


class MachineLearningModel:
    def __init__(self):
        self.dataset: Dict[DataNecessityType, DataSetObject] = {}

    def update_data(self,
            training_input: Optional[List[Any]] = None,
            training_output: Optional[List[Any]] = None,
            testing_input: Optional[List[Any]] = None,
            testing_output: Optional[List[Any]] = None) -> None:
        ignore_on_dataset_init_errors: Tuple = (DataSetValuesIsNoneTypeError, )

        training_data_necessity_type = DataNecessityType.TRAINING
        training_dataset = DataSetObject(
            input_data=training_input,
            output_data=training_output,
            data_necessity_type=training_data_necessity_type,
            ignore_errors=ignore_on_dataset_init_errors
        )
        if training_dataset.error_occurred_on_init is None:
            self.dataset[training_data_necessity_type] = training_dataset

        testing_data_necessity_type = DataNecessityType.TESTING
        testing_dataset = DataSetObject(
            input_data=testing_input,
            output_data=testing_output,
            data_necessity_type=testing_data_necessity_type,
            ignore_errors=ignore_on_dataset_init_errors
        )
        if testing_dataset.error_occurred_on_init is None:
            self.dataset[testing_data_necessity_type] = testing_dataset

    def predict(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement predict function")

    def loss_function(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement loss_function")
