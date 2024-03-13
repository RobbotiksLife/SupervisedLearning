import matplotlib.pyplot as plt

PLOT_TITLE = 'Gradient Descent Optimization'


def loss_function_x_cubed(x):
    return x ** 3


def gradient_x_cubed(x):
    return 3 * x


def define_plot(function, x_values, plot_filename, title=PLOT_TITLE):
    plt.figure(figsize=(10, 6))
    border: int = 2
    r = range(int(min(x_values))-border, int(max(x_values))+border)
    plt.plot(r, [function(x) for x in r], label='Loss')
    plt.scatter(x_values, [function(x) for x in x_values], color='blue')
    plt.plot(x_values, [function(x) for x in x_values], color='red')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    if plot_filename:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    else:
        plt.show()


def gradient_descent(epochs, loss_function, gradient, learning_rate, plot_filename=None, title=PLOT_TITLE):
    x = 10  # Initial guess for x
    x_values = []
    loss_values = []
    gradient_values = []

    for epoch in range(epochs):
        grad = gradient(x)
        x = x - learning_rate * grad
        loss = loss_function(x)
        print(f"Epoch {epoch + 1}: x = {x}, Loss = {loss} | Gradient = {grad}(After: {learning_rate * grad})")

        x_values.append(x)
        loss_values.append(loss)
        gradient_values.append(grad)
    define_plot(loss_function, x_values, plot_filename, title=title)


def compare_pow_learning(epochs, learning_rate, pow_1, pow_2):
    define_print_str = lambda pow, L: f"learning with x^{pow} and learning_rate={L}"
    define_plot_title_str = lambda pow, L: f"Gradient Descent Optimization x^{pow}(with L={L})"
    define_plot_filename_str = lambda pow, L: f"gradient_descent_plot_comparison_pow{pow}_L{L}.png"
    print(define_print_str(pow=pow_1, L=learning_rate))
    gradient_descent(
        epochs=epochs,
        loss_function=lambda x: x ** pow_1,
        gradient=lambda x: pow_1 * x,
        learning_rate=learning_rate,
        plot_filename=define_plot_filename_str(pow_1, learning_rate),
        title=define_plot_title_str(pow_1, learning_rate)
    )

    print(define_print_str(pow=pow_2, L=learning_rate))
    gradient_descent(
        epochs=epochs,
        loss_function=lambda x: x ** pow_2,
        gradient=lambda x: pow_2 * x,
        learning_rate=learning_rate,
        plot_filename=define_plot_filename_str(pow_2, learning_rate),
        title=define_plot_title_str(pow_2, learning_rate)
    )


if __name__ == '__main__':
    compare_pow_learning(
        epochs=25,
        learning_rate=0.25,
        pow_1=2,
        pow_2=10
    )
