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

# LRM.b0 = 20000
# %% test learning
loss_history = LRM.learn(
    epochs=7500,
    learning_factor=0.01,
    data_necessity_type=DataNecessityType.TRAINING
)
calculate_loss_print_and_save_data("plot_linear_regression_after_learning", LRM)
# %% visualize  history of learning
LRM.plot_loss_history(loss_history)

# %% setting to zero
LRM.b0 = 0
LRM.b1 = 0
calculate_loss_print_and_save_data("plot_linear_regression_after_resetting_params_to_zero", LRM)

# %% define mathematically ideal params
LRM.define_mathematically_ideal_params()
calculate_loss_print_and_save_data("plot_linear_regression_mathematically_ideal_params", LRM)


data_necessity_type = DataNecessityType.TRAINING
dataset_with_defined_data_necessity_type = LRM.dataset.get(data_necessity_type)
n, Ex, Ey, Exx, Exy = LRM.define_learning_data(dataset_with_defined_data_necessity_type)

# %% visualize loss function surface and history of learning
LRM.visualize_3d_b0_l1_loss_dependency(
    function=lambda b1, b0: LRM.loss_function(b0, b1, data_necessity_type),
    history_points=loss_history[::50],
    name="Loss Surface Visualization"
)

LRM.visualize_3d_b0_l1_loss_dependency(
    function=lambda b1, b0: LRM.b0_derivative_on_loss(b0, b1, n, Ex, Ey),
    history_points=loss_history[::50],
    name="B1 Derivation Surface Visualization"
)

LRM.visualize_3d_b0_l1_loss_dependency(
    function=lambda b1, b0: LRM.b1_derivative_on_loss(b0, b1, n, Exx, Ex, Exy),
    history_points=loss_history[:4:1],
    name="B0 Derivation Surface Visualization"
)

