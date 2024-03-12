from LinearRegressionModel import *
import pandas as pd
import matplotlib.pyplot as plt


def define_linear_regression_plot(x_data, y_data, b0, b1, name, format="png", range=range(0, 12)):
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
    epochs=7500,
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

