from models.SimpleKNearestNeighbors import *

def plot_prediction(k=3, data_points_num = 50, format = "png"):
    data_points_num_each_class = int(data_points_num / 2)
    # Generate random data points for class 0
    np.random.seed(0)
    X_class0 = np.random.randn(data_points_num_each_class, 2) * 1.5 + np.array([3, 3])

    # Generate random data points for class 1
    X_class1 = np.random.randn(data_points_num_each_class, 2) * 2.5 + np.array([8, 8])

    # Combine the data points and labels
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([np.zeros(data_points_num_each_class), np.ones(data_points_num_each_class)])


    # Plot the data points
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title('Randomly Generated Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'k_nearest_neighbors_dataset_len{data_points_num}.{format}')

    # Initialize and train the KNN classifier
    classifier = KNearestNeighbors(k=k)
    classifier.fit(X, y)

    # Plot decision regions
    plt.figure(figsize=(10, 6))
    plot_decision_regions(X, y, classifier)
    plt.title('K Nearest Neighbors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(f'k_nearest_neighbors_performance_len{data_points_num}_k{k}.{format}')

if __name__ == '__main__':
    plot_prediction(
        k=3,
        data_points_num=250
    )


