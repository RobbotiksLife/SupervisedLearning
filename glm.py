import pickle
import random

from models.GeneralizedLinearRegressionModel import *
import pandas as pd

from models.SimpleGeneralizedLinearRegressionModel import *


def linear_prediction_function(xs: [float], b_params: [float]):
    return sum([(bn * (xs[i - 1] ** i) if i != 0 else bn) for (i, bn) in enumerate(b_params)])

def first_derivative_of_linear_prediction_function(respect_to_bn: int, xs: List[float]):
    return (xs[respect_to_bn - 1] ** respect_to_bn) if respect_to_bn != 0 else 1

def first_derivative_of_linear_prediction_function_as_exponent_LP(respect_to_bn: int, xs: List[float], b_params: [float]):
    return first_derivative_of_linear_prediction_function(
        respect_to_bn=respect_to_bn, xs=xs
    ) * math.exp(linear_prediction_function(
        xs=xs, b_params=b_params
    ))


def custom_x_y_dataset_data_test():
    # %% load data
    dataset = pd.read_csv('datasets/custom_x_y_dataset.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    # x_train_2 = [x ** 2 for x in x_train_1]
    y_train = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = SimpleGeneralizedLinearRegressionModel(
        start_b0=0,
        start_b1=1
    )
    LRM.update_data(
        training_input={
            "x_train_1": x_train_1
        },
        training_output=y_train
    )

    # # %% test learning
    loss_history = LRM.learn(
        epochs=1000,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=1
    )
    LRM.plot_loss_history(loss_history, model_description_str="_glm_custom_x_y_dataset_data_test_normal_exp")

    LRM.plot_performance(
        x_data=x_train_1,
        y_data=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_glm_custom_x_y_dataset_data_test_normal_exp",
        padding_interval=3
    )

    n = 0.5
    LRM.plot_b0_dependency_from_loss(
        r=float_range(LRM.b0 - n, LRM.b0 + n, 0.01),
        name="b0_loss_dependency_derivative_glm_custom_x_y_dataset_data_test_normal_exp"
    )

def custom_x_y_dataset_data_test_2():
    # %% load data
    dataset = pd.read_csv('datasets/custom_x_y_dataset_2.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    # x_train_2 = [x ** 2 for x in x_train_1]
    y_train = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = SimpleGeneralizedLinearRegressionModel(
        start_b0=0,
        start_b1=1
    )
    LRM.update_data(
        training_input={
            "x_train_1": x_train_1
        },
        training_output=y_train
    )

    # # %% test learning
    loss_history = LRM.learn(
        epochs=1050,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=1
    )
    LRM.plot_loss_history(loss_history, model_description_str="_glm_custom_x_y_dataset_data_test_2_normal_exp")

    LRM.plot_performance(
        x_data=x_train_1,
        y_data=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_glm_custom_x_y_dataset_data_test_2_normal_exp",
        padding_interval=3,
        step_quality=0.01
    )

    n = 0.5
    LRM.plot_b0_dependency_from_loss(
        r=float_range(LRM.b0 - n, LRM.b0 + n, 0.01),
        name="b0_loss_dependency_derivative_glm_custom_x_y_dataset_data_test_2_normal_exp"
    )

def sales_data():
    # %% load data
    dataset = pd.read_csv('datasets/Salary_Data.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    # x_train_2 = [x ** 2 for x in x_train_1]
    y_train = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = SimpleGeneralizedLinearRegressionModel(
        start_b0=0,
        start_b1=1
    )
    LRM.update_data(
        training_input={
            "x_train_1": x_train_1
        },
        training_output=y_train
    )

    # # %% test learning
    loss_history = LRM.learn(
        epochs=1000,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=1
    )
    LRM.plot_loss_history(loss_history, model_description_str="_glm_sales_data_x_y_test_normal_exp")

    LRM.plot_performance(
        x_data=x_train_1,
        y_data=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_glm_sales_data_x_y_test_normal_exp",
        padding_interval=0,
        step_quality=0.01
    )


if __name__ == '__main__':
    sales_data()
    custom_x_y_dataset_data_test()
    custom_x_y_dataset_data_test_2()


