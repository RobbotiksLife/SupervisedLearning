import math
from decimal import Decimal

from models.ExponentialFamilyDistribution import *
from models.MachineLearningModel import *

import plotly.graph_objs as go
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import sigmoid

class SimpleLogisticRegressionModel(MachineLearningModel):
    def __init__(self, start_b0: float = 0, start_b1: float = 1):
        super().__init__()
        self.b0: float = start_b0
        self.b1: float = start_b1

    def learn(self, epochs: int, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING, epoch_history_save_interval=10, learning_factor: float = None):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        loss_history = []
        e = 0
        while e < epochs:
            epoch_loss = -self.log_likelihood_function(
                b0=self.b0,
                b1=self.b1,
                data_necessity_type=data_necessity_type
            )

            if (e+1) % epoch_history_save_interval == 0:
                # print(f"Epoch: {e} -> b0: {self.b0}, b1: {self.b1}")
                print(f"Epoch: {e+1} | Loss: {epoch_loss} -> b0: {self.b0}, b1: {self.b1}")
                loss_history.append({
                    "B0": self.b0,
                    "B1": self.b1,
                    "epoch_loss": epoch_loss
                })

            # b0_edit = 0
            # b1_edit = 0

            b0_derivative = self.b0_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b0_second_derivative = self.b0_second_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            self.b0 += b0_derivative * (1 / abs(b0_second_derivative))
            # b0_edit += b0_derivative * (Decimal(1) / abs(b0_second_derivative))

            b1_derivative = self.b1_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b1_second_derivative = self.b1_second_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            self.b1 += b1_derivative * (1 / abs(b1_second_derivative))
            # b1_edit += b1_derivative * (Decimal(1) / abs(b1_second_derivative))  # learning_factor

            # self.b0 += b0_edit
            # self.b1 += b1_edit

            e += 1
        return loss_history

    @staticmethod
    def b0_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2*(y*math.exp(b0+b1*x) - math.exp(2*b0+2*b1*x))  # Normal d/db0(d(y, m))
            # result += -2*(y - math.exp(b0+b1*x))  # Poisson d/db0(d(y, m))

            # lpexp = math.exp(b0+b1*x)
            # result += (1/(1+lpexp) if y == 1 else -(lpexp/(1+lpexp)))

            # lpexp = np.exp(b0 + b1 * x)
            # sigmoid_value = 1 / (1 + lpexp)
            # result = sigmoid_value if y == 1 else -lpexp*sigmoid_value

            lpexp = np.exp(b0 + b1 * x)
            result += (1/(1+lpexp) if y == 1 else -(lpexp/(1+lpexp)))

            # lpexp = (Decimal(b0) + Decimal(b1) * Decimal(x)).exp()
            # result = Decimal(1) / (Decimal(1) + lpexp) if y == 1 else -lpexp / (Decimal(1) + lpexp)
        return result

    @staticmethod
    def b1_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2 * (y * x * math.exp(b0 + b1 * x) - x * math.exp(2 * b0 + 2 * b1 * x))  # Normal d/db1(d(y, m))
            # result += -2 * x * (y - math.exp(b0 + b1 * x))  # Poisson d/db1(d(y, m))

            # lpexp = math.exp(b0 + b1 * x)
            # result += (x / (1 + lpexp) if y == 1 else -((x*lpexp) / (1 + lpexp)))

            lpexp = np.exp(b0 + b1 * x)
            result += (x / (1 + lpexp) if y == 1 else -((x*lpexp) / (1 + lpexp)))

            # lpexp = (Decimal(b0) + Decimal(b1) * Decimal(x)).exp()
            # result = Decimal(x) / (Decimal(1) + lpexp) if y == 1 else -(Decimal(x) * lpexp) / (Decimal(1) + lpexp)
        return result


    @staticmethod
    def b0_second_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2*(y*math.exp(b0+b1*x) - 2*math.exp(2*b0+2*b1*x))  # Normal d/db0^2(d(y, m))
            # result += 2 * math.exp(b0 + b1 * x)  # Poisson d/db0^2(d(y, m))

            # lpexp = math.exp(b0 + b1 * x)
            # result += -(lpexp / math.pow((1+lpexp), 2))

            lpexp = np.exp(b0 + b1 * x)
            result += -(lpexp / np.power((1+lpexp), 2))

            # lpexp = (Decimal(b0) + Decimal(b1) * Decimal(x)).exp()
            # result += -lpexp / ((Decimal(1) + lpexp) ** Decimal(2))
        return result

    @staticmethod
    def b1_second_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2*(y*math.pow(x, 2)*math.exp(b0+b1*x) - 2*math.pow(x, 2)*math.exp(2*b0+2*b1*x))  # Normal d/db1^2(d(y, m))
            # result += 2 * math.pow(x, 2) * math.exp(b0 + b1 * x)  # Poisson d/db1^2(d(y, m))

            # lpexp = math.exp(b0 + b1 * x)
            # result += -((math.pow(x, 2) * lpexp) / math.pow((1 + lpexp), 2))

            lpexp = np.exp(b0 + b1 * x)
            result += -((np.power(x, 2) * lpexp) / np.power((1 + lpexp), 2))

            # lpexp = (Decimal(b0) + Decimal(b1) * Decimal(x)).exp()
            # result += -((Decimal(1) ** 2) * lpexp) / ((Decimal(1) + lpexp) ** Decimal(2))
        return result

    # @staticmethod
    # def b1_derivative_on_loss(b0: float, b1: float, n: int, Exx: float, Ex: float, Exy: float):
    #     return 2/n * (b1*Exx + b0*Ex - Exy)

    def predict(self, x):
        return self.prediction_function(
            x=x,
            b0=self.b0,
            b1=self.b1
        )

    def log_likelihood_function(self, b0: float, b1: float, data_necessity_type: DataNecessityType):
        e = 0
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        for i in range(dataset_with_defined_data_necessity_type.data_len):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]
            # predicted_y = self.prediction_function(
            #     x=x,
            #     b0=b0,
            #     b1=b1
            # )
            # e += (predicted_y - y) ** 2  # Normal d(y, m)
            # e += 2 * (y * (math.log(y/predicted_y) - 1) + predicted_y)  # Poisson d(y, m)
            # value = predicted_y if y == 1 else 1-predicted_y


            # lp = self.linear_prediction_function(x, b0, b1)
            # lpexp = math.exp(lp)
            # lnlpexpplusone = math.log(1+lpexp)
            # value = (lp-lnlpexpplusone) if y == 1 else (-lnlpexpplusone)

            lp = self.linear_prediction_function(x, b0, b1)
            lpexp = np.exp(lp)
            lnlpexpplusone = np.log(1+lpexp)
            value = (lp-lnlpexpplusone) if y == 1 else (-lnlpexpplusone)

            # lp_decimal = Decimal(str(lp))
            # lpexp = (-lp_decimal).exp()
            # lnlpexpplusone = (Decimal('1') + lpexp).ln()
            # value = (lp_decimal - lnlpexpplusone) if y == 1 else (-lnlpexpplusone)

            e += value
        return e

    def prediction_function(self, x: float, b0: float, b1: float):
        lp = self.linear_prediction_function(x, b0, b1)
        lpexp = np.exp(lp)
        result = lpexp / (1 + lpexp)
        # lpexp = Decimal(str(math.exp(lp)))
        # lp_decimal = Decimal(str(lp))
        # lpexp = Decimal(1) + (-lp_decimal).exp()  # Using the exp() method of Decimal
        # result = lpexp / (Decimal(1) + lpexp)
        if result == 1:
            print("WTF?")
        return result

    @staticmethod
    def linear_prediction_function(x: float, b0: float, b1: float):
        # return Decimal(b0) + (Decimal(b1) * Decimal(x))
        return b0 + b1 * x

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

    # /// ----------------------------- PLOTTING ----------------------------- ///

    @staticmethod
    def plot_log_likelihood_history(history_list, save_path="", model_description_str=""):
        # Extract values
        epochs = [i for i in range(len(history_list))]
        epoch_loss = [point['epoch_loss'] for point in history_list]
        # Plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, epoch_loss, marker='o', color='b', label='Loss per Epoch')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_path}plot_loss_history{model_description_str}.png')

    # @staticmethod
    # def define_linear_regression_plot(x_data, y_data, b0, b1, name, format="png", range=range(0, 12)):
    #     plt.figure()
    #     plt.scatter(x_data, y_data, color='red')
    #     plt.plot(range, [b1*x + b0 for x in range], color="black")
    #     plt.title(name)
    #     plt.xlabel('Experience in years')
    #     plt.ylabel('Salary')
    #     plt.savefig(f'{name}.{format}')


    def plot_performance(self, x, y, b0, b1, name, padding_interval=5, format="png", step_quality=0.1):
        # min_point = int(min(min(x_data), min(y_data)))
        # max_point = int(max(max(x_data), max(y_data)))
        min_point = min(x) - padding_interval
        max_point = max(x) + padding_interval
        # r = range(min_point, max_point)
        r = float_range(min_point, max_point, step=step_quality)
        plt.figure(figsize=(8, 6))

        x_train_1_0 = [x[i] for i in range(len(y)) if y[i] == 0]
        x_train_1_1 = [x[i] for i in range(len(y)) if y[i] == 1]

        # Plotting the dataset
        plt.scatter(x_train_1_0, [0] * len(x_train_1_0), color='blue', label='Y = 0')
        plt.scatter(x_train_1_1, [1] * len(x_train_1_1), color='orange', label='Y = 1')

        plt.plot(r, [self.prediction_function(x, b0, b1) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')


    def plot_b0_derivative(self, r, b1=None, name="b0_derivative", format="png"):  # float_range(-100, 100, 0.5)
        if b1 is None:
            b1 = self.b1
        # r = range(min_point, max_point)
        # r = float_range(min_point, max_point, 0.5)
        plt.figure()
        plt.plot(r, [self.b0_derivative_on_loss(
            b0=x,
            b1=b1,
            dataset_with_defined_data_necessity_type=self.dataset.get(DataNecessityType.TRAINING)
        ) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')

    def plot_b0_dependency_from_log_likelihood_function(self, r, b1=None, name="b0_derivative", format="png"):  # float_range(-100, 100, 0.5)
        if b1 is None:
            b1 = self.b1
        # r = range(min_point, max_point)
        # r = float_range(min_point, max_point, 0.5)
        plt.figure()
        plt.plot(r, [self.log_likelihood_function(
            b0=x,
            b1=b1,
            data_necessity_type=DataNecessityType.TRAINING
        ) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')


    def plot_loss_by_b0(self, min_point, max_point, b0, b1, name, format="png"):
        r = range(min_point, max_point)
        plt.figure()
        plt.plot(r, [self.log_likelihood_function(b0=x, b1=b1, data_necessity_type=DataNecessityType.TRAINING) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')