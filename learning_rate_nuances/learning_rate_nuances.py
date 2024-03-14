import re
import matplotlib.pyplot as plt
import numpy as np

PLOT_TITLE = 'Gradient Descent Optimization'


def loss_function_x_cubed(x):
    return x ** 3


def gradient_x_cubed(x):
    return 3 * x


def define_plot(function, x_values, plot_filename, plot_path="", title=PLOT_TITLE):
    plt.figure(figsize=(10, 6))
    border: int = 2
    min_value = int(min(x_values))-border
    max_value = int(max(x_values))+border
    graf_quality = (max_value-min_value)/len(x_values)
    r = np.arange(min_value, max_value, step=graf_quality)
    plt.plot(r, [function(x) for x in r], label='Loss')
    plt.scatter(x_values, [function(x) for x in x_values], color='blue')
    plt.plot(x_values, [function(x) for x in x_values], color='red')
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    if plot_filename:
        plt.savefig(f"{plot_path}{plot_filename}")
        print(f"Plot saved to {plot_path}{plot_filename}")
    else:
        plt.show()


def gradient_descent(epochs, loss_function, gradient, learning_rate, plot_filename=None, title=PLOT_TITLE):
    x = 10  # Initial guess for x
    x_values = []
    loss_values = []
    gradient_values = []

    for epoch in range(epochs):
        grad = gradient(x)
        L = learning_rate(x)
        x = x - L * grad
        loss = loss_function(x)
        print(f"Epoch {epoch + 1}: x = {x}, Loss = {loss} | Gradient = {grad}(After: {L * grad})")

        x_values.append(x)
        loss_values.append(loss)
        gradient_values.append(grad)
    define_plot(loss_function, x_values, plot_filename, plot_path="learning_rate_nuances/", title=title)


def compare_pow_learning(epochs, learning_rate_1, learning_rate_2, pow_1, pow_2, constant_learning_rate=False):
    defined_learning_rate_str_value = lambda L: f" (with L0={L(0.001)})" if constant_learning_rate else " (with adoptive learning rate)"
    define_print_str = lambda pow, L: f"learning with x^{pow}{defined_learning_rate_str_value(L)}"
    define_plot_title_str = lambda pow, L: f"Gradient Descent Optimization x^{pow}{defined_learning_rate_str_value(L)}"
    define_plot_filename_str = lambda pow, L: f"gradient_descent_plot_comparison_pow{pow}{ re.sub(r'[()]', '', defined_learning_rate_str_value(L).replace(' ', '_'))}.png"
    print(define_print_str(pow=pow_1, L=learning_rate_1))
    gradient_descent(
        epochs=epochs,
        loss_function=lambda x: x ** pow_1,
        gradient=lambda x: pow_1 * (x ** (pow_1-1)),
        learning_rate=learning_rate_1,
        plot_filename=define_plot_filename_str(pow_1, learning_rate_1),
        title=define_plot_title_str(pow_1, learning_rate_1)
    )

    print(define_print_str(pow=pow_2, L=learning_rate_2))
    gradient_descent(
        epochs=epochs,
        loss_function=lambda x: x ** pow_2,
        gradient=lambda x: pow_2 * (x ** (pow_2-1)),
        learning_rate=learning_rate_2,
        plot_filename=define_plot_filename_str(pow_2, learning_rate_2),
        title=define_plot_title_str(pow_2, learning_rate_2)
    )


def compare_pow_learning_newton(epochs, pow_1, pow_2):
    compare_pow_learning(
        epochs=epochs,
        learning_rate_1=lambda x: 1/(pow_1 * (pow_1-1) * (x ** (pow_1-2))),
        learning_rate_2=lambda x: 1/(pow_2 * (pow_2-1) * (x ** (pow_2-2))),
        pow_1=pow_1,
        pow_2=pow_2,
        constant_learning_rate=False
    )


if __name__ == '__main__':
    # Learn with static learning rate
    # compare_pow_learning(
    #     epochs=1000,
    #     learning_rate_1=lambda x: 0.25,
    #     learning_rate_2=lambda x: 0.025,
    #     pow_1=2,
    #     pow_2=3,
    #     constant_learning_rate=True
    # )

    # Learn using newton algorithm do define adaptive learning rate
    compare_pow_learning_newton(
        epochs=1000,
        pow_1=2,
        pow_2=10
    )
