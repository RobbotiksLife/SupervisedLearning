from enum import Enum
from typing import List, Any, Dict, Optional, Tuple
import math


class PredictionType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

class DataNecessityType(Enum):
    TRAINING = "TRAINING"
    TESTING = "TESTING"

class DataType:
    def __init__(self, prediction_type: PredictionType, data_necessity_type: DataNecessityType):
        self.prediction_type = prediction_type
        self.data_necessity_type = data_necessity_type

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



class LinearRegressionModel(MachineLearningModel):
    def __init__(self, start_b0: float = 0, start_b1: float = 0):
        super().__init__()
        self.b0: float = start_b0
        self.b1: float = start_b1

    def learn(self, epochs: int, learning_factor: float = 0.01, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        n, Ex, Ey, Exx, Exy = self.define_learning_data(dataset_with_defined_data_necessity_type)
        loss_history = []
        e = 0
        while e <= epochs:
            epoch_loss = self.loss_function(
                b0=self.b0,
                b1=self.b1,
                data_necessity_type=data_necessity_type
            )

            if e % 10 == 0:
                print(f"Epoch: {e} | Loss: {epoch_loss}")
            loss_history.append({
                "B0": self.b0,
                "B1": self.b1,
                "epoch_loss": epoch_loss
            })

            self.b0 = self.b0 - self.b0_derivative_on_loss(n, Ex, Ey) * learning_factor
            self.b1 = self.b1 - self.b1_derivative_on_loss(n, Exx, Ex, Exy) * learning_factor

            e += 1
        return loss_history

    def b0_derivative_on_loss(self, n: int, Ex: float, Ey: float):
        return 2/n * (self.b1*Ex + n*self.b0 - Ey)

    def b1_derivative_on_loss(self, n: int, Exx: float, Ex: float, Exy: float):
        return 2/n * (self.b1*Exx + self.b0*Ex - Exy)

    def predict(self, x):
        return self.prediction_function(
            x=x,
            b0=self.b0,
            b1=self.b1
        )

    def define_mathematically_ideal_params(self, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        if dataset_with_defined_data_necessity_type is None:
            raise ValueError("Unable to define dataset with provided data_necessity_type")

        n, Ex, Ey, Exx, Exy = self.define_learning_data(dataset_with_defined_data_necessity_type)

        self.b1 = (n*Exy - Ex*Ey) / (n*Exx - Ex ** 2)
        self.b0 = (Ey - self.b1*Ex) / n

    def loss_function(self, b0: float, b1: float, data_necessity_type: DataNecessityType):
        e = 0
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        for i in range(dataset_with_defined_data_necessity_type.data_len):
            predicted_y = self.prediction_function(
                x=dataset_with_defined_data_necessity_type.input_data[i],
                b0=b0,
                b1=b1
            )
            e = (predicted_y - dataset_with_defined_data_necessity_type.output_data[i]) ** 2
        return e / dataset_with_defined_data_necessity_type.data_len

    @staticmethod
    def define_learning_data(dataset_with_defined_data_necessity_type: DataSetObject):
        n = dataset_with_defined_data_necessity_type.data_len
        Ex = sum([dataset_with_defined_data_necessity_type.input_data[i] for i in range(n)])
        Ey = sum([dataset_with_defined_data_necessity_type.output_data[i] for i in range(n)])
        Exx = sum([dataset_with_defined_data_necessity_type.input_data[i] ** 2 for i in range(n)])
        Exy = sum([dataset_with_defined_data_necessity_type.input_data[i]
                   * dataset_with_defined_data_necessity_type.output_data[i] for i in range(n)])
        return n, Ex, Ey, Exx, Exy

    @staticmethod
    def prediction_function(x: float, b0: float, b1: float):
        return b0 + b1*x

    def prediction_function_str(self, b0: Optional[float] = None, b1: Optional[float] = None):
        if b0 is None or b1 is None:
            return self._prediction_function_str(b0=self.b0, b1=self.b1)
        else:
            return self._prediction_function_str(b0=b0, b1=b1)

    @staticmethod
    def _prediction_function_str(b0: float, b1: float):
        return f"y = {b0} + {b1}*x"


import pandas as pd
import matplotlib.pyplot as plt


def define_linear_regression_plot(x_data, y_data, b0, b1, name, format="png", range = range(0, 12)):
    plt.figure()
    plt.scatter(x_data,y_data,color='red')
    plt.plot(range, [b1*x + b0 for x in r], color="black")
    plt.title('Salary vs Experience(Train set)')
    plt.xlabel('Experience in years')
    plt.ylabel('Salary')
    plt.savefig(f'{name}.{format}')

# %% load data
dataset = pd.read_csv('Salary_Data.csv')
x_train = dataset.iloc[:, 0].values
y_train = dataset.iloc[:, 1].values

LRM = LinearRegressionModel()
LRM.update_data(
    training_input=list(x_train),
    training_output=list(y_train)
)
n, Ex, Ey, Exx, Exy = LRM.define_learning_data(LRM.dataset.get(DataNecessityType.TRAINING))

loss_history = LRM.learn(
    epochs=10000,
    learning_factor=0.01,
    data_necessity_type=DataNecessityType.TRAINING
)
print(f"loss_value_after_resetting_params_to_zero: {loss_history[-1]} ({LRM.prediction_function_str()})")

# %% plot the data and the model
r = range(9000, 12000)
plt.figure()
plt.plot(r, [LRM.loss_function(0, b1, DataNecessityType.TRAINING) for b1 in r], color="black")
plt.title('Salary vs Experience(Train set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary')
plt.savefig('plot_b1_derivative_on_loss.png')


import plotly.graph_objs as go


def show_loss_history_3d(data_list):
    x = [point['B0'] for point in data_list]
    y = [point['B1'] for point in data_list]
    z = [point['epoch_loss'] for point in data_list]

    # Create traces for scatter plot
    scatter_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        )
    )

    # Create traces for arrows
    arrow_traces = []
    for i in range(len(x) - 1):
        arrow_trace = go.Scatter3d(
            x=[x[i], x[i + 1]],
            y=[y[i], y[i + 1]],
            z=[z[i], z[i + 1]],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='none'
        )
        arrow_traces.append(arrow_trace)

    # Create the figure
    fig = go.Figure(data=[scatter_trace] + arrow_traces)

    # Set labels and title
    fig.update_layout(scene=dict(
                        xaxis_title='B0',
                        yaxis_title='B1',
                        zaxis_title='epoch_loss'),
                      title='3D Scatter Plot with Arrows')

    # Show the plot
    fig.show()

def show_loss_history(data_list):
    # Extract values
    epochs = [i for i in range(len(data_list))]
    epoch_loss = [point['epoch_loss'] for point in data_list]
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, epoch_loss, marker='o', color='b', label='Loss per Epoch')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_loss_history.png')

# show_loss_history_3d(loss_history)

show_loss_history(loss_history)


LRM.b0 = 0
LRM.b1 = 0
loss_value_after_resetting_params_to_zero = LRM.loss_function(
    b0=LRM.b0,
    b1=LRM.b1,
    data_necessity_type=DataNecessityType.TRAINING
)
print(f"loss_value_after_resetting_params_to_zero: {loss_value_after_resetting_params_to_zero} ({LRM.prediction_function_str()})")

LRM.define_mathematically_ideal_params()
loss_value_with_mathematically_ideal_params = LRM.loss_function(
    b0=LRM.b0,
    b1=LRM.b1,
    data_necessity_type=DataNecessityType.TRAINING
)
print(f"loss_value_with_mathematically_ideal_params: {loss_value_with_mathematically_ideal_params} ({LRM.prediction_function_str()})")


# print(loss_history)

