import numpy as np
from sklearn.utils import shuffle


class LinearSVM:

    def __init__(self, regularization_param):
        """
        Initialize the model by setting the regularization parameter
        and a boolean variable for our trained weights.
        """
        self.regularization_param = regularization_param
        self.trained_weights = None

    def add_bias_term(self, features):
        """
        Add intercept 1 to each training example for bias b
        """
        n_samples = features.shape[0]
        ones = np.ones((n_samples, 1))
        return np.concatenate((ones, features), axis=1)

    def compute_cost(self, weights, features, labels) -> float:
        """
        Compute the value of the cost function
        """
        n_samples = features.shape[0]

        # Compute hinge loss
        predictions = np.dot(features, weights).flatten()
        distances = 1 - labels * predictions
        hinge_losses = np.maximum(0, distances)

        # Compute sum of the individual hinge losses
        sum_hinge_loss = np.sum(hinge_losses) / n_samples

        # Compute entire cost
        cost = (1 / 2) * np.dot(weights.T, weights) + self.regularization_param * sum_hinge_loss

        return float(cost)

    def compute_gradient(self, weights, features, labels) -> np.ndarray:
        """
        Compute the gradient, needed for training
        """
        predictions = np.dot(features, weights)
        distances = 1 - labels * predictions
        n_samples, n_feat = features.shape
        sub_gradients = np.zeros((1, n_feat))

        for idx, dist in enumerate(distances):
            if max(0, dist) == 0:
                sub_gradients += weights.T
            else:
                sub_grad = weights.T - (self.regularization_param * features[idx] * labels[idx])
                sub_gradients += sub_grad

        # Sum up and divide by the number of samples
        avg_gradient = sum(sub_gradients) / len(labels)

        return avg_gradient

    def train(self, train_features, train_labels, n_epochs, learning_rate=0.01, batch_size=1):
        """
        Train the model with stochastic gradient descent using the
        specified number of epochs, learning rate and batch size.
        """
        # Add bias term to features
        train_features = self.add_bias_term(train_features)

        # Initalize weight vector
        n_samples, n_feat = train_features.shape
        weights = np.zeros(n_feat)[:, np.newaxis]

        # Train the model for a certain number of epochs
        for epoch in range(n_epochs):
            features, labels = shuffle(train_features, train_labels)
            features, labels = train_features, train_labels
            start, end = 0, batch_size
            while end <= len(labels):  # Training loop over the dataset
                batch = features[start:end]
                batch_labels = labels[start:end]

                grad = self.compute_gradient(weights, batch, batch_labels)
                update = (learning_rate * grad)[:, np.newaxis]
                weights = weights - update
                start, end = end, end + batch_size

            current_cost = self.compute_cost(weights, features, labels)
            print(f"Epoch {epoch + 1}, cost: {current_cost}")

        # Set the trained weights to allow making predictions
        self.trained_weights = weights

    def predict(self, test_features) -> np.ndarray:
        """
        Predict labels for new test features.
        Raises ValueError if model has not been trained yet.
        """
        test_features = self.add_bias_term(test_features)
        if self.trained_weights is None:
            raise ValueError("You haven't trained the SVM yet!")

        predicted_labels = []
        n_samples = test_features.shape[0]
        for idx in range(n_samples):
            prediction = np.sign(np.dot(self.trained_weights.T, test_features[idx]))
            predicted_labels.append(prediction)

        return np.array(predicted_labels)


# Compute some values to make sure the cost is computed correctly
# I calculated the values for this example by hand first
svm = LinearSVM(regularization_param=1)
weights = np.array([1, 2])[:, np.newaxis]
features = np.array([[0.5], [2.5]])
new_features = svm.add_bias_term(features)

labels = np.array([-1, +1])
assert svm.compute_cost(weights, new_features, labels) == 4.
gradient = svm.compute_gradient(weights, new_features, labels[:, np.newaxis])



# Initialize a new SVM and train it on the given toy dataset
regularization_param = 100
lr = 0.000001
svm = LinearSVM(regularization_param)
trained_weights = svm.train(features_train, labels_train, n_epochs=10, learning_rate=lr)