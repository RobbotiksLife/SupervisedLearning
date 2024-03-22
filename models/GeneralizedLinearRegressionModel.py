import copy
import math

import matplotlib.pyplot as plt

from models.ExponentialFamilyDistribution import *
from models.MachineLearningModel import *
from utils.extrection import *


class GeneralizedLinearRegressionModel(MachineLearningModel):
    def __init__(self, distribution: NormalDistribution, link_function=lambda x: x, derivative_of_linear_link_prediction_function_with_respect_to_bn=None):
        super().__init__()
        self.link_function = link_function
        self.distribution = distribution
        self.b_params = []

        if derivative_of_linear_link_prediction_function_with_respect_to_bn is None:
            self.derivative_of_linear_link_prediction_function_with_respect_to_bn = lambda respect_to_bn, xs, b_params: self.first_derivative_of_linear_prediction_function(respect_to_bn=respect_to_bn, xs=xs)
        else:
            self.derivative_of_linear_link_prediction_function_with_respect_to_bn = lambda respect_to_bn, xs, b_params: derivative_of_linear_link_prediction_function_with_respect_to_bn(respect_to_bn=respect_to_bn, xs=xs, b_params=b_params)

    def update_params_recording_dataset(self, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        number_of_x_params = dataset_with_defined_data_necessity_type.shape[0]
        self.b_params = [0.0 for _ in range(number_of_x_params+1)]

    def update_link_function(self, function):
        self.link_function = lambda x: function(x)

    def learn(self, epochs: int, learning_factor: Optional[float] = None, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING, epoch_history_save_interval=10):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        loss_history = []
        e = 0
        while e <= epochs:
            epoch_loss = self.loss_function(
                data_necessity_type=data_necessity_type,
                b_params=self.b_params
            )

            if e % epoch_history_save_interval == 0:
                print(f"Epoch: {e} | Loss: {epoch_loss}")
                loss_history.append({
                    "params": self.b_params,
                    "epoch_loss": epoch_loss
                })

            for bn_param_index in range(len(self.b_params)):
                self.b_params[bn_param_index] -= self.derivative_on_loss_with_respect_to_bn(
                    respect_to_bn=bn_param_index,
                    dataset_with_defined_data_necessity_type=dataset_with_defined_data_necessity_type,
                    b_params=self.b_params
                ) * ((1 / self.second_derivative_on_loss_with_respect_to_bn(
                    respect_to_bn=bn_param_index,
                    dataset_with_defined_data_necessity_type=dataset_with_defined_data_necessity_type
                )) if learning_factor is None else learning_factor)

            e += 1
        return loss_history

    def derivative_on_loss_with_respect_to_bn(self, respect_to_bn: int, dataset_with_defined_data_necessity_type: DataSetObject, b_params: [float]):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            xs = extract_values_by_index(
                data=dataset_with_defined_data_necessity_type.input_data,
                index_to_extract=i
            )

            result += self.d_function_derivative(
                respect_to_bn=respect_to_bn,
                xs=xs,
                y=y,
                b_params=b_params
            )
        return result

    @staticmethod
    def second_derivative_on_loss_with_respect_to_bn(respect_to_bn: int, dataset_with_defined_data_necessity_type: DataSetObject):
        # n = dataset_with_defined_data_necessity_type.data_len
        # if respect_to_bn != 0:
        #     sum_der_result = 0
        #     for i in range(n):
        #         xs = extract_values_by_index(
        #             data=dataset_with_defined_data_necessity_type.input_data,
        #             index_to_extract=i
        #         )
        #
        #         sum_der_result += (xs[respect_to_bn - 1] ** (2 * respect_to_bn))
        # else:
        #     sum_der_result = n
        # return (2 / n) * sum_der_result
        return 1

    # def one_derivative_calculation_iteration_on_loss_with_respect_to_bn(self, respect_to_bn: int, xs: List[float], y: float, b_params: [float]):
    #     derivative_of_function_inside = self.first_derivative_of_linear_prediction_function(
    #         respect_to_bn=respect_to_bn,
    #         xs=xs
    #     )
    #     return ((self.prediction_function(
    #         xs=xs,
    #         b_params=b_params
    #     ) - y) * derivative_of_function_inside)

    def d_function_derivative(self, respect_to_bn: int, xs: List[float], y: float, b_params: [float]):  # , y, m
        return -2 * (y * (
            self.derivative_of_linear_link_prediction_function_with_respect_to_bn(
                respect_to_bn=respect_to_bn,
                xs=xs,
                b_params=b_params
            )
        ) - (
            self.distribution.beta_function_derivative(
                m=self.link_function(self.linear_prediction_function(xs=xs, b_params=b_params)),
                teta_function_derivative=None
            )
        ))

    def predict(self, xs: List[float]):
        return self.prediction_function(
            xs=xs,
            b_params=self.b_params
        )

    def r_squared(self, bn: int, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        params_without_bn = copy.deepcopy(self.b_params)
        params_without_bn[bn] = 0
        loss_without_bn = self.loss_function(
            data_necessity_type=data_necessity_type,
            b_params=params_without_bn
        )
        loss_with_bn = self.loss_function(
            data_necessity_type=data_necessity_type,
            b_params=self.b_params
        )
        return (loss_without_bn - loss_with_bn) / loss_without_bn

    def r_squared_str(self, bn: int, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING, show_function: bool = True):
        return f"R^2 for feature {self.define_feature_for_bn(bn=bn-1, data_necessity_type=data_necessity_type) if bn!=0 else 'BASE'}" \
               f" as B{bn}{f' in {self.prediction_function_str()}' if show_function else ''} is {self.r_squared(bn=bn, data_necessity_type=data_necessity_type)}"

    def define_feature_for_bn(self, bn: int, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING) -> str:
        return list(self.dataset.get(data_necessity_type).keys)[bn]

    def show_all_r_squared(self, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING, show_function: bool = True):
        for param_index in range(len(self.b_params)):
            print(self.r_squared_str(
                bn=param_index,
                data_necessity_type=data_necessity_type,
                show_function=show_function
            ))

    def __str__(self):
        return f"<GeneralizedLinearRegressionModel | {self.prediction_function_str()}>"

    def loss_function(self, data_necessity_type: DataNecessityType, b_params: [float]):
        e = 0
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            point_y = dataset_with_defined_data_necessity_type.output_data[i]
            predicted_y = self.prediction_function(
                xs=extract_values_by_index(
                    data=dataset_with_defined_data_necessity_type.input_data,
                    index_to_extract=i
                ),
                b_params=b_params
            )
            # e += (predicted_y - point_y) ** 2
            e += self.distribution.d_function(point_y, predicted_y)
        return e  # e / n

    # def derivative_of_linear_link_prediction_function_with_respect_to_bn(self, xs: [float], b_params: [float]):
    #     return self.link_function(self.linear_prediction_function(xs=xs, b_params=b_params))

    def prediction_function(self, xs: [float], b_params: [float]):
        return self.link_function(self.linear_prediction_function(xs=xs, b_params=b_params))

    @staticmethod
    def linear_prediction_function(xs: [float], b_params: [float]):
        return sum([(bn*(xs[i-1] ** i) if i != 0 else bn) for (i, bn) in enumerate(b_params)])

    @staticmethod
    def first_derivative_of_linear_prediction_function(respect_to_bn: int, xs: List[float]):
        return (xs[respect_to_bn - 1] ** respect_to_bn) if respect_to_bn != 0 else 1

    def prediction_function_str(self, params: [float] = None):
        if params is None:
            return self._prediction_function_str(params=self.b_params)
        else:
            return self._prediction_function_str(params=params)

    @staticmethod
    def _prediction_function_str(params: [float]):
        function_str = "y = LinkFunction("
        for i, param in enumerate(params):
            if i == 0:
                function_str += f"{param}"
            else:
                function_str += f" + {param}*D{f'^{i}' if i != 1 else ''}"
        return function_str + ")"

    def update_data(self,
            training_input: Optional[Dict[str, List[float]]] = None,
            testing_input: Optional[Dict[str, List[float]]] = None,
            training_output: Optional[List[float]] = None,
            testing_output: Optional[List[float]] = None) -> None:
        self._update_data(
            training_input=training_input,
            testing_input=testing_input,
            training_output=training_output,
            testing_output=testing_output
        )
        self.update_params_recording_dataset()

    # @staticmethod
    # def plot_loss_history(history_list, save_path=""):
    #     # Extract values
    #     epochs = [i for i in range(len(history_list))]
    #     epoch_loss = [point['epoch_loss'] for point in history_list]
    #     # Plotting the data
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(epochs, epoch_loss, marker='o', color='b', label='Loss per Epoch')
    #     plt.title('Loss per Epoch')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(f'{save_path}plot_loss_history.png')
    #
    # @staticmethod
    # def plot_performance(x_data, y_data, b0, b1, name, format="png"):
    #     # min_point = int(min(min(x_data), min(y_data)))
    #     # max_point = int(max(max(x_data), max(y_data)))
    #     min_point = int(min(x_data))
    #     max_point = int(max(x_data))
    #     r = range(min_point, max_point)
    #     plt.figure()
    #     plt.scatter(x_data, y_data, color='red')
    #     plt.plot(r, [b1*x + b0 for x in r], color="black")
    #     plt.title(name)
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.savefig(f'{name}.{format}')
    #
    # def plot_performance(self, x_data, y_data, dimension, name, format="png", range=range(0, 12)):
    #     plt.figure()
    #     plt.scatter(x_data, y_data, color='red')
    #     plt.plot(range, [self.prediction_function(xs=xs,b_params=b_params) for x in range], color="black")
    #     plt.title(name)
    #     plt.xlabel('Experience in years')
    #     plt.ylabel('Salary')
    #     plt.savefig(f'{name}.{format}')


    @staticmethod
    def plot_performance(x_data, y_data, b0, b1, name, format="png"):
        # min_point = int(min(min(x_data), min(y_data)))
        # max_point = int(max(max(x_data), max(y_data)))
        min_point = int(min(x_data))
        max_point = int(max(x_data))
        r = range(min_point, max_point)
        plt.figure()
        plt.scatter(x_data, y_data, color='red')
        plt.plot(r, [math.exp(b1*x + b0) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')