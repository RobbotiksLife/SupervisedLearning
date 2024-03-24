import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

from models.SimpleLogisticRegressionModel import *


def plot_dataset(x, y, name="logistic_regression_dataset", format="png"):
    plt.figure(figsize=(8, 6))
    # Separate points based on y values
    x_train_1_0 = [x[i] for i in range(len(y)) if y[i] == 0]
    x_train_1_1 = [x[i] for i in range(len(y)) if y[i] == 1]

    # Plotting the dataset
    plt.scatter(x_train_1_0, [0] * len(x_train_1_0), color='blue', label='Y = 0')
    plt.scatter(x_train_1_1, [1] * len(x_train_1_1), color='orange', label='Y = 1')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of the Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name}.{format}")



def test_with_custom_logistic_dataset_1():
    # %% load data
    dataset = pd.read_csv('datasets/custom_logistic_dataset_1.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    y_train = list(dataset.iloc[:, 1].values)
    print(x_train_1)
    print(y_train)

    # plot_dataset(x=x_train_1, y=y_train)

    # %% define data
    LRM = SimpleLogisticRegressionModel(
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
    LRM.plot_log_likelihood_history(
        loss_history,
        model_description_str="_logistic_regression_custom_dataset_1_learning_history"
    )

    LRM.plot_performance(
        x=x_train_1,
        y=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_logistic_regression_custom_dataset_1",
        padding_interval=0.5,
        step_quality=0.01
    )


def test_with_custom_logistic_dataset_2():
    # %% load data
    dataset = pd.read_csv('datasets/custom_logistic_dataset_2.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    y_train = list(dataset.iloc[:, 1].values)
    print(x_train_1)
    print(y_train)

    # plot_dataset(x=x_train_1, y=y_train)

    # %% define data
    LRM = SimpleLogisticRegressionModel(
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
    LRM.plot_log_likelihood_history(
        loss_history,
        model_description_str="_logistic_regression_custom_dataset_2_learning_history"
    )

    LRM.plot_performance(
        x=x_train_1,
        y=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_logistic_regression_custom_dataset_2",
        padding_interval=0.5,
        step_quality=0.01
    )

def test_with_iris_dataset():
    # Load data
    iris = datasets.load_iris()

    # Prepare the data
    x = [x[0] for x in iris["data"][:, 3:]]  # petal width
    y = list((iris["target"] == 2).astype(int))  # 1 if Iris virginica, else 0

    # generated_set_len = 5
    # random_indexes = [random.randint(0, len(x) - 1) for _ in range(generated_set_len)]
    # x_train_1 = [x[index] for index in random_indexes]
    # y_train = [y[index] for index in random_indexes]

    x_train_1 = x
    y_train = y

    print(x_train_1)
    print(y_train)

    plot_dataset(x=x_train_1, y=y_train)

    # %% define data
    LRM = SimpleLogisticRegressionModel(
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
    LRM.plot_log_likelihood_history(
        loss_history,
        model_description_str="_logistic_regression_iris_dataset_learning_history"
    )

    LRM.plot_performance(
        x=x_train_1,
        y=y_train,
        b0=LRM.b0,
        b1=LRM.b1,
        name="performance_logistic_regression_iris_dataset",
        padding_interval=0.5,
        step_quality=0.01
    )


if __name__ == '__main__':
    # test_with_custom_logistic_dataset_1()
    # test_with_custom_logistic_dataset_2()
    test_with_iris_dataset()
