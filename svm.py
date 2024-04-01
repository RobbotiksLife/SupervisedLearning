from models.SupportVectorMachine import *
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_features, data_targets = make_blobs(n_samples=600, centers=2, n_features=2, random_state=42)
    # The function outputs targets 0 and 1 so we need to convert targets 0 to -1
    transformed_data_targets = [-1 if t == 0 else +1 for t in data_targets]

    SLSVM = SimpleSupportVectorMachine(
        regularization_param=1
    )

    dataset = [LinearSVMPoint(x=data_features[i][0], y=data_features[i][1], class_value=transformed_data_targets[i]) for i in range(len(transformed_data_targets))]

    loss_history = SLSVM.learn(
        epochs=100,
        dataset=dataset,
        epoch_history_save_interval=1
    )

    data_features_x = [point[0] for point in data_features]
    data_features_y = [point[1] for point in data_features]
    data_features_x_min = min(data_features_x)
    data_features_x_max = max(data_features_x)
    data_features_y_min = min(data_features_y)
    # Define a range for x1 values
    x1_values = np.linspace(data_features_x_min, data_features_x_max, 100)

    # Calculate corresponding x2 values based on the hyperplane equation
    x2_values = (-SLSVM.w0 - SLSVM.w1 * x1_values) / SLSVM.w2

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(data_features[:, 0], data_features[:, 1], c=transformed_data_targets)
    plt.plot(x1_values, x2_values, label='Hyperplane', color='red')
    plt.title("Toy dataset")
    plt.ylabel("Feature 2")
    plt.xlabel("Feature 1")
    plt.savefig(f"svm_performance_c_{SLSVM.regularization_param}.png")

