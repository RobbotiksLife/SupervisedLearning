from LinearRegressionModel import *
import pandas as pd


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
LRM.define_linear_regression_plot(
    x_data=LRM.dataset.get(DataNecessityType.TRAINING).input_data,
    y_data=LRM.dataset.get(DataNecessityType.TRAINING).output_data,
    b0=LRM.b0,
    b1=LRM.b1,
    name="loss_value_after_resetting_params_to_zero"
)

LRM.visualize_loss_surface(data_necessity_type=DataNecessityType.TRAINING, history_points=loss_history[::50])

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
LRM.define_linear_regression_plot(
    x_data=LRM.dataset.get(DataNecessityType.TRAINING).input_data,
    y_data=LRM.dataset.get(DataNecessityType.TRAINING).output_data,
    b0=LRM.b0,
    b1=LRM.b1,
    name="loss_value_with_mathematically_ideal_params"
)