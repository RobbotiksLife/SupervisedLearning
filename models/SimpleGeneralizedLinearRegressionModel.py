import math

from models.ExponentialFamilyDistribution import *
from models.MachineLearningModel import *

import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimpleGeneralizedLinearRegressionModel(MachineLearningModel):
    def __init__(self, start_b0: float = 0, start_b1: float = 1):
        super().__init__()
        self.b0: float = start_b0
        self.b1: float = start_b1

    def learn(self, epochs: int, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING, epoch_history_save_interval=10, learning_factor: float = None):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        loss_history = []
        e = 0
        while e <= epochs:
            epoch_loss = self.loss_function(
                b0=self.b0,
                b1=self.b1,
                data_necessity_type=data_necessity_type
            )

            if e % epoch_history_save_interval == 0:
                print(f"Epoch: {e} | Loss: {epoch_loss} -> b0: {self.b0}, b1: {self.b1}")
                loss_history.append({
                    "B0": self.b0,
                    "B1": self.b1,
                    "epoch_loss": epoch_loss
                })

            b0_edit = 0
            b1_edit = 0

            b0_derivative = self.b0_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b0_second_derivative = self.b0_second_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b0_edit += b0_derivative * (1 / abs(b0_second_derivative))

            b1_derivative = self.b1_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b1_second_derivative = self.b1_second_derivative_on_loss(self.b0, self.b1, dataset_with_defined_data_necessity_type)
            b1_edit += b1_derivative * (1 / abs(b1_second_derivative))  # learning_factor

            self.b0 -= b0_edit
            self.b1 -= b1_edit

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
            result += -2*(y - math.exp(b0+b1*x))  # Poisson d/db0(d(y, m))
        return result

    @staticmethod
    def b1_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2 * (y * x * math.exp(b0 + b1 * x) - x * math.exp(2 * b0 + 2 * b1 * x))  # Normal d/db1(d(y, m))
            result += -2 * x * (y - math.exp(b0 + b1 * x))  # Poisson d/db1(d(y, m))
        return result


    @staticmethod
    def b0_second_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2*(y*math.exp(b0+b1*x) - 2*math.exp(2*b0+2*b1*x))  # Normal d/db0^2(d(y, m))
            result += 2 * math.exp(b0 + b1 * x)  # Poisson d/db0^2(d(y, m))
        return result

    @staticmethod
    def b1_second_derivative_on_loss(b0: float, b1: float, dataset_with_defined_data_necessity_type):
        result = 0
        n = dataset_with_defined_data_necessity_type.data_len
        for i in range(n):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]

            # result += -2*(y*math.pow(x, 2)*math.exp(b0+b1*x) - 2*math.pow(x, 2)*math.exp(2*b0+2*b1*x))  # Normal d/db1^2(d(y, m))
            result += 2 * math.pow(x, 2) * math.exp(b0 + b1 * x)  # Poisson d/db1^2(d(y, m))
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

    def loss_function(self, b0: float, b1: float, data_necessity_type: DataNecessityType):
        e = 0
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        for i in range(dataset_with_defined_data_necessity_type.data_len):
            y = dataset_with_defined_data_necessity_type.output_data[i]
            x = dataset_with_defined_data_necessity_type.input_data[0][i]
            predicted_y = self.prediction_function(
                x=x,
                b0=b0,
                b1=b1
            )
            # e += (predicted_y - y) ** 2  # Normal d(y, m)
            e += 2 * (y * (math.log(y/predicted_y) - 1) + predicted_y)  # Poisson d(y, m)
        return e

    @staticmethod
    def prediction_function(x: float, b0: float, b1: float):
        return math.exp(b0 + b1*x)

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
    def visualize_3d_b0_b1_loss_dependency(
            function,
            history_points,
            name,
            squared_range_value_b0=500000,
            squared_range_value_b1=50000,
            quality_factor=0.01,
            save_path=''
    ):
        # Generate b0, b1 grid
        b0_range = np.arange(-squared_range_value_b0, squared_range_value_b0, squared_range_value_b0*quality_factor)
        b1_range = np.arange(-squared_range_value_b1, squared_range_value_b1, squared_range_value_b1*quality_factor)
        b0_grid, b1_grid = np.meshgrid(b0_range, b1_range)

        # Compute loss for each combination of b0 and b1
        loss_values = np.zeros_like(b0_grid)
        for i in tqdm(range(len(b0_range))):
            for j in range(len(b1_range)):
                loss_values[j, i] = function(b0=b0_range[i], b1=b1_range[j])

        # Find the minimum point
        min_indices = np.where(np.abs(loss_values) < 10000)  # 100150
        min_b0_values = b0_range[min_indices[1]]
        min_b1_values = b1_range[min_indices[0]]
        # min_loss_values = loss_values[min_indices]
        min_loss_values = []
        for i in range(len(min_b0_values)):
            min_loss_values.append(function(b0=min_b0_values[i], b1=min_b1_values[i]))

        # Create surface plot
        fig = go.Figure(data=[go.Surface(x=b0_range, y=b1_range, z=loss_values)])

        # Plot the points close to 0
        fig.add_trace(go.Scatter3d(
            x=min_b0_values,
            y=min_b1_values,
            z=min_loss_values,
            mode='markers',
            marker=dict(size=5, color='yellow'),
            name='Close to 0')
        )

        # Add history points
        for i in range(len(history_points)):
            point = history_points[i]
            b0 = point['B0']
            b1 = point['B1']
            loss = function(b0=point['B0'], b1=point['B1'])
            fig.add_trace(go.Scatter3d(x=[b0], y=[b1], z=[loss], mode='markers', marker=dict(size=5, color='red'), name=f'HP(E:{i})'))
            if i > 0:
                prev_point = history_points[i-1]
                fig.add_trace(go.Scatter3d(
                    x=[prev_point['B0'], b0],
                    y=[prev_point['B1'], b1],
                    z=[function(b0=prev_point['B0'], b1=prev_point['B1']), loss],
                    mode='lines',
                    line=dict(width=2, color='red'),
                    name=f'M(E:{i})'
                ))

            # Connect history point to surface with lines
            fig.add_trace(go.Scatter3d(
                x=[b0, b0],
                y=[b1, b1],
                z=[loss, 0],
                mode='lines',
                line=dict(color='blue', width=1),
                showlegend=False
            ))

        fig.update_layout(
            title=name,
            scene=dict(
                xaxis_title='b0',
                yaxis_title='b1',
                zaxis_title='Loss'
            )
        )
        # fig.show()
        # Save the figure to HTML file
        fig.write_html(f"{save_path}{name}.html")

    @staticmethod
    def plot_loss_history(history_list, save_path="", model_description_str=""):
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


    @staticmethod
    def plot_performance(x_data, y_data, b0, b1, name, padding_interval=5, format="png", step_quality=0.1):
        # min_point = int(min(min(x_data), min(y_data)))
        # max_point = int(max(max(x_data), max(y_data)))
        min_point = int(min(x_data)) - padding_interval
        max_point = int(max(x_data))+1 + padding_interval
        # r = range(min_point, max_point)
        r = float_range(min_point, max_point, step=step_quality)
        plt.figure()
        plt.scatter(x_data, y_data, color='red')
        plt.plot(r, [math.exp(b1*x + b0) for x in r], color="black")
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

    def plot_b0_dependency_from_loss(self, r, b1=None, name="b0_derivative", format="png"):  # float_range(-100, 100, 0.5)
        if b1 is None:
            b1 = self.b1
        # r = range(min_point, max_point)
        # r = float_range(min_point, max_point, 0.5)
        plt.figure()
        plt.plot(r, [self.loss_function(
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
        plt.plot(r, [self.loss_function(b0=x, b1=b1, data_necessity_type=DataNecessityType.TRAINING) for x in r], color="black")
        plt.title(name)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'{name}.{format}')