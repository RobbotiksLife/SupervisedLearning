from LinearRegressionModel import *
import pandas as pd

def calculate_loss_print_and_save_data(name: str, LRM: LinearRegressionModel, data_necessity_type=DataNecessityType.TRAINING):
    loss_value_with_mathematically_ideal_params = LRM.loss_function(
        b0=LRM.b0,
        b1=LRM.b1,
        data_necessity_type=data_necessity_type
    )
    print(
        f"{name}: {loss_value_with_mathematically_ideal_params} ({LRM.prediction_function_str()})")
    LRM.define_linear_regression_plot(
        x_data=LRM.dataset.get(data_necessity_type).input_data,
        y_data=LRM.dataset.get(data_necessity_type).output_data,
        b0=LRM.b0,
        b1=LRM.b1,
        name=name
    )

# %% load data
dataset = pd.read_csv('Salary_Data.csv')
x_train = dataset.iloc[:, 0].values
y_train = dataset.iloc[:, 1].values

# %% define data
LRM = LinearRegressionModel()
LRM.update_data(
    training_input=list(x_train),
    training_output=list(y_train)
)

# %% test learning
loss_history = LRM.learn(
    epochs=7500,
    learning_factor=0.001,
    data_necessity_type=DataNecessityType.TRAINING
)
calculate_loss_print_and_save_data("plot_linear_regression_after_learning", LRM)

# %% setting to zero
LRM.b0 = 0
LRM.b1 = 0
calculate_loss_print_and_save_data("plot_linear_regression_after_resetting_params_to_zero", LRM)

# %% define mathematically ideal params
LRM.define_mathematically_ideal_params()
calculate_loss_print_and_save_data("plot_linear_regression_mathematically_ideal_params", LRM)

# %% visualize loss function surface and history of learning
# LRM.visualize_loss_surface(data_necessity_type=DataNecessityType.TRAINING, history_points=loss_history[::50])

# %% visualize  history of learning
LRM.plot_loss_history(loss_history)

LRM.try_visualize_loss_surface_b1_derivative(data_necessity_type=DataNecessityType.TRAINING, history_points=loss_history[::10])