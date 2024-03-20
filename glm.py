import pickle
import random

from models.GeneralizedLinearRegressionModel import *
import pandas as pd


def sales_data():
    # %% load data
    dataset = pd.read_csv('datasets/Salary_Data.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    # x_train_2 = [x ** 2 for x in x_train_1]
    y_train = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = GeneralizedLinearRegressionModel()
    LRM.update_data(
        training_input={
            "x_train_1": x_train_1
        },
        training_output=y_train
    )

    # # %% test learning
    # loss_history = LRM.learn(
    #     epochs=40,
    #     learning_factor=None,
    #     data_necessity_type=DataNecessityType.TRAINING,
    #     epoch_history_save_interval=1
    # )
    # # %% visualize  history of learning
    # print(f"Function after learning: {LRM.prediction_function_str()}")
    # LRM.show_all_r_squared()
    # LRM.plot_loss_history(loss_history, "")
    # LRM.plot_performance(
    #     x_data=x_train_1,
    #     y_data=y_train,
    #     b0=LRM.b_params[0],
    #     b1=LRM.b_params[1],
    #     name="performance"
    # )


if __name__ == '__main__':
    sales_data()


