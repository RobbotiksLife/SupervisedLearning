from models.MachineLearningModel import *

import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm
import matplotlib.pyplot as plt


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

            self.b0 = self.b0 - self.b0_derivative_on_loss(self.b0, self.b1, n, Ex, Ey) * learning_factor
            self.b1 = self.b1 - self.b1_derivative_on_loss(self.b0, self.b1, n, Exx, Ex, Exy) * learning_factor

            e += 1
        return loss_history

    @staticmethod
    def b0_derivative_on_loss(b0: float, b1: float, n: int, Ex: float, Ey: float):
        return 2/n * (b1*Ex + n*b0 - Ey)

    @staticmethod
    def b1_derivative_on_loss(b0: float, b1: float, n: int, Exx: float, Ex: float, Exy: float):
        return 2/n * (b1*Exx + b0*Ex - Exy)

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
            e += (predicted_y - dataset_with_defined_data_necessity_type.output_data[i]) ** 2
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

    # /// ----------------------------- PLOTTING ----------------------------- ///

    @staticmethod
    def visualize_3d_b0_l1_loss_dependency(
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
    def plot_loss_history(history_list, save_path=""):
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
        plt.savefig(f'{save_path}plot_loss_history.png')

    @staticmethod
    def define_linear_regression_plot(x_data, y_data, b0, b1, name, format="png", range=range(0, 12)):
        plt.figure()
        plt.scatter(x_data, y_data, color='red')
        plt.plot(range, [b1*x + b0 for x in range], color="black")
        plt.title(name)
        plt.xlabel('Experience in years')
        plt.ylabel('Salary')
        plt.savefig(f'{name}.{format}')


    def define_plot_test(self, x_data, y_data, b0, b1, name, format="png", range=range(0, 12)):
        plt.figure()
        plt.scatter(x_data, y_data, color='red')
        plt.plot(range, [b1*x + b0 for x in range], color="black")
        plt.title(name)
        plt.xlabel('Experience in years')
        plt.ylabel('Salary')
        plt.savefig(f'{name}.{format}')