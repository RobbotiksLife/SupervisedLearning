import pickle
import random

from models.LinearRegressionModel import *
import pandas as pd


def sales_data():
    # %% load data
    dataset = pd.read_csv('datasets/Salary_Data.csv')
    x_train_1 = list(dataset.iloc[:, 0].values)
    # x_train_2 = [x ** 2 for x in x_train_1]
    y_train = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = LinearRegressionModel()
    LRM.update_data(
        training_input=[
            x_train_1
        ],
        training_output=y_train
    )

    # %% test learning
    loss_history = LRM.learn(
        epochs=40,
        learning_factor=None,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=1
    )
    # %% visualize  history of learning
    print(f"Function after learning: {LRM.prediction_function_str()}")
    LRM.show_all_r_squared()
    LRM.plot_loss_history(loss_history, "")
    LRM.plot_performance(
        x_data=x_train_1,
        y_data=y_train,
        b0=LRM.b_params[0],
        b1=LRM.b_params[1],
        name="performance"
    )

def walmart_sales():
    # %% load data
    dataset = pd.read_csv('datasets/Walmart_sales.csv')
    # x predictions
    date = list(dataset.iloc[:, 2].values)
    holiday_flag = list(dataset.iloc[:, 3].values)
    temperature = list(dataset.iloc[:, 4].values)
    fuel_price = list(dataset.iloc[:, 5].values)
    cpi = list(dataset.iloc[:, 6].values)
    unemployment = list(dataset.iloc[:, 7].values)
    # y prediction
    weekly_sales = list(dataset.iloc[:, 2].values)

    # %% define data
    LRM = LinearRegressionModel()
    LRM.update_data(
        training_input=[
            # date,
            temperature,
            holiday_flag,
            fuel_price,
            cpi,
            unemployment
        ],
        training_output=weekly_sales
    )

    # %% learning
    loss_history = LRM.learn(
        epochs=1000,
        learning_factor=None,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=50
    )
    # %% visualize  history of learning
    print(f"Function after learning: {LRM.prediction_function_str()}")
    LRM.show_all_r_squared()
    # LRM.plot_loss_history(loss_history, "")
    # LRM.plot_performance(
    #     x_data=date,
    #     y_data=weekly_sales,
    #     b0=LRM.b_params[0],
    #     b1=LRM.b_params[1],
    #     name="performance"
    # )


def ice_cream_sales():
    # %% load data
    dataset = pd.read_csv('datasets/Ice Cream Sales - temperatures.csv')
    # x predictions
    temperature = list(dataset.iloc[:, 0].values)
    # y prediction
    sales = list(dataset.iloc[:, 1].values)

    # %% define data
    LRM = LinearRegressionModel()
    LRM.update_data(
        training_input=[
            temperature,
        ],
        training_output=sales
    )

    # %% learning
    loss_history = LRM.learn(
        epochs=250,
        learning_factor=None,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=50
    )
    # %% visualize  history of learning
    print(f"Function after learning: {LRM.prediction_function_str()}")
    LRM.show_all_r_squared()
    LRM.plot_loss_history(loss_history, "")
    LRM.plot_performance(
        x_data=temperature,
        y_data=sales,
        b0=LRM.b_params[0],
        b1=LRM.b_params[1],
        name="performance"
    )


def students_performance():
    # %% load data
    dataset = pd.read_csv('datasets/Student_Performance.csv')
    # x predictions
    HoursStudied = list(dataset.iloc[:, 0].values)
    PreviousScores = list(dataset.iloc[:, 1].values)
    ExtracurricularActivities = [True if x == "Yes" else False for x in list(dataset.iloc[:, 2].values)]
    SleepHours = list(dataset.iloc[:, 3].values)
    SampleQuestionPapersPracticed = list(dataset.iloc[:, 4].values)
    # y prediction
    PerformanceIndex = list(dataset.iloc[:, 5].values)

    # %% define data
    LRM = LinearRegressionModel()
    LRM.update_data(
        training_input={
            "PreviousScores": PreviousScores[:5],
            # "HoursStudied": HoursStudied[:5],
            # "RandomData": [random.random() for _ in range(len(PerformanceIndex))][:5],
            # "ExtracurricularActivities": ExtracurricularActivities[:5],
            # "SleepHours": SleepHours[:5],
            # "SampleQuestionPapersPracticed": SampleQuestionPapersPracticed[:5],
        },
        training_output=PerformanceIndex[:5]
    )

    # %% learning
    loss_history = LRM.learn(
        epochs=150,
        learning_factor=None,
        data_necessity_type=DataNecessityType.TRAINING,
        epoch_history_save_interval=10
    )
    LRM.plot_loss_history(loss_history, "")


    # Load model
    LRM: Optional[LinearRegressionModel] = None
    with open('students_performance_lrm.pkl', 'rb') as f:
        LRM = pickle.load(f)

    # %% visualize  history of learning
    print(f"Function after learning: {LRM.prediction_function_str()}")
    print(f"Function loss: {LRM.loss_function(data_necessity_type=DataNecessityType.TRAINING, b_params=LRM.b_params)}")
    LRM.show_all_r_squared(show_function=False)
    # LRM.plot_performance(
    #     x_data=PreviousScores,
    #     y_data=PerformanceIndex,
    #     b0=LRM.b_params[0],
    #     b1=LRM.b_params[1],
    #     name="performance"
    # )

    name = "performance_t"
    format = "png"
    data_necessity_type: DataNecessityType = DataNecessityType.TRAINING
    dataset_with_defined_data_necessity_type = LRM.dataset.get(data_necessity_type)
    # min_point = int(min(x_data))
    # max_point = int(max(x_data))
    # r = range(min_point, max_point)
    x_values = PreviousScores[:5]
    y_values = PerformanceIndex[:5]

    r = range(len(x_values))
    plt.figure()
    plt.scatter(x_values, y_values, color='red')
    plt.plot(x_values, [LRM.prediction_function(
        xs=extract_values_by_index(
            data=dataset_with_defined_data_necessity_type.input_data,
            index_to_extract=i
        ),
        b_params=LRM.b_params
    ) for i in r], color="black")
    plt.title(name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'{name}.{format}')

    with open('students_performance_lrm.pkl', 'wb') as f:
        pickle.dump(LRM, f)


if __name__ == '__main__':
    # sales_data()
    # ice_cream_sales()
    students_performance()
