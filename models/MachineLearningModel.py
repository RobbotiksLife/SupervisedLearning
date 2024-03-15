from typing import Dict
from models.DataSetObject import *


class MachineLearningModel:
    def __init__(self):
        self.dataset: Dict[DataNecessityType, DataSetObject] = {}

    def _update_data(self,
            training_input: Optional[Dict[str, List[float]]] = None,
            testing_input: Optional[Dict[str, List[float]]] = None,
            training_output: Optional[List[float]] = None,
            testing_output: Optional[List[float]] = None) -> None:
        ignore_on_dataset_init_errors: Tuple = (DataSetValuesIsNoneTypeError, )
        if training_input is not None and testing_input is not None and len(training_input.values()) == len(testing_input.values()):
            raise ValueError("training_output shape should be same as testing_output")

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
