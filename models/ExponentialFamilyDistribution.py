import math
import matplotlib.pyplot as plt
from utils.utils import *


class ExponentialFamilyDistribution:
    def teta_function(self, m):
        raise NotImplementedError("Subclasses must implement teta_function function")

    def alfa_function(self, f):
        raise NotImplementedError("Subclasses must implement alfa_function function")

    def beta_function(self, m):
        raise NotImplementedError("Subclasses must implement beta_function function")

    def constant_function(self, y, f):
        raise NotImplementedError("Subclasses must implement constant_function function")

    def t_function(self, y, m):
        raise NotImplementedError("Subclasses must implement t_function function")

    # def t_function_derivative(self, y, m):
    #     raise NotImplementedError("Subclasses must implement t_function function")

    def d_function(self, y, m):
        return 2 * (self.t_function(y, y) - self.t_function(y, m))

    def count(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement count function")

    def count_log(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement count_log function")


class NormalDistribution(ExponentialFamilyDistribution):
    def teta_function(self, m):
        return m

    def alfa_function(self, f):
        return math.pow(f, 2)

    def beta_function(self, m):
        return math.pow(self.teta_function(m), 2) / 2

    def constant_function(self, y, f):
        return -(math.pow(y, 2)/(2*math.pow(f, 2))) - math.log(f * math.sqrt(2*math.pi))

    def t_function(self, y, m):
        return y * self.teta_function(m) - self.beta_function(m)

    def d_function(self, y, m):
        return 2 * (self.t_function(y, y) - self.t_function(y, m))

    def beta_function_derivative(self, m, teta_function_derivative):
        return self.teta_function(m)

    def count(self, y, std, mean):
        return math.exp(self.count_log(y, std, mean))

    def count_log(self, y, std, mean):
        return self.alfa_function(std) * self.t_function(y, mean) + self.constant_function(y, std)

    def plot_count_function(self, std=1.0, mean=0.0, path="", format="png", range_y=range(0, 12)):
        name = f"normal_distribution_plot_std_{std}_mean_{mean}"
        plt.figure()
        plt.plot(range_y, [self.count(
            y=y,
            std=std,
            mean=mean
        ) for y in range_y], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_t_function(self, mean=0.0, path="", format="png", range_y=range(0, 12)):
        name = f"normal_distribution_t_function_plot_with_mean_{mean}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_y, [self.t_function(
            y=y,
            m=mean
        ) for y in range_y], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_t_function_mean_change_2d(self, y=0.0, path="", format="png", range_mean=range(0, 12)):
        name = f"normal_distribution_t_function_plot_with_static_y_{y}_mean_form_{range_mean[0]}_to_{range_mean[-0]}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_mean, [self.t_function(
            y=y,
            m=mean
        ) for mean in range_mean], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')


class ExponentialDistribution(ExponentialFamilyDistribution):
    def teta_function(self, lambda_p):
        return -lambda_p

    def alfa_function(self, f):
        return 1

    def beta_function(self, lambda_p):
        return -math.log(-self.teta_function(lambda_p))

    def constant_function(self, y, f):
        return 0

    def t_function(self, y, lambda_p):
        return y * self.teta_function(lambda_p) - self.beta_function(lambda_p)

    def d_function(self, y, lambda_p):
        return 2 * (self.t_function(y, y) - self.t_function(y, lambda_p))

    def count(self, y, lambda_p):
        return math.exp(self.count_log(y, lambda_p))

    def count_log(self, y, lambda_p):
        return self.alfa_function(1) * self.t_function(y, lambda_p) + self.constant_function(y, 1)

    def plot_count_function(self, lambda_p=1.0, path="", format="png", range_y=range(0, 12)):
        name = f"exponential_distribution_plot_with_lambda_p_{lambda_p}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_y, [self.count(
            y=y,
            lambda_p=lambda_p
        ) for y in range_y], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_t_function(self, lambda_p=1.0, path="", format="png", range_y=range(0, 12)):
        name = f"exponential_distribution_t_function_plot_with_lambda_p_{lambda_p}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_y, [self.t_function(
            y=y,
            lambda_p=lambda_p
        ) for y in range_y], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_t_function_lambda_p_change_2d(self, y=0.0, path="", format="png", range_lambda_p=range(0, 12)):
        name = f"exponential_distribution_t_function_plot_with_static_y_{y}_lambda_p_from_{range_lambda_p[0]}_to_{range_lambda_p[-0]}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_lambda_p, [self.t_function(
            y=y,
            lambda_p=lambda_p
        ) for lambda_p in range_lambda_p], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')


    def plot_d_function_lambda_p_change_2d(self, y=0.0, path="", format="png", range_lambda_p=range(0, 12)):
        name = f"exponential_distribution_d_function_plot_with_static_y_{y}_lambda_p_from_{range_lambda_p[0]}_to_{range_lambda_p[-0]}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_lambda_p, [self.d_function(
            y=y,
            lambda_p=lambda_p
        ) for lambda_p in range_lambda_p], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')


class PoissonDistribution(ExponentialFamilyDistribution):
    def teta_function(self, lambda_p):
        return math.log(lambda_p)  # ln(l)

    def alfa_function(self, f):
        return 1

    def beta_function(self, lambda_p):
        # return math.exp(self.teta_function(lambda_p))
        return lambda_p  # exp(teta_function(lambda_p)) -> exp(ln(l)) -> l

    def constant_function(self, y, f):
        return -math.log(math.factorial(y))  # -ln(y!)

    def t_function(self, y, lambda_p):
        return y * self.teta_function(lambda_p) - self.beta_function(lambda_p)

    def d_function(self, y, lambda_p):
        return 2 * (self.t_function(y, y) - self.t_function(y, lambda_p))

    def count(self, y, lambda_p):
        return math.exp(self.count_log(y, lambda_p))

    def count_log(self, y, lambda_p):
        return self.alfa_function(None) * self.t_function(y, lambda_p) + self.constant_function(y, None)

    def plot_count_function(self, lambda_p=1.0, path="", format="png", range_y=range(0, 12)):
        name = f"poisson_distribution_plot_with_lambda_{lambda_p}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        res = [self.count(
            y=y,
            lambda_p=lambda_p
        ) for y in range_y]
        plt.scatter(range_y, res, color="black")
        plt.plot(range_y, res, color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_t_function_lambda_p_change_2d(self, y=0.0, path="", format="png", range_lambda_p=range(0, 12)):
        name = f"poisson_distribution_t_function_plot_with_static_y_{y}_lambda_from_{range_lambda_p[0]}_to_{range_lambda_p[-0]}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_lambda_p, [self.t_function(
            y=y,
            lambda_p=lambda_p
        ) for lambda_p in range_lambda_p], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')

    def plot_d_function_lambda_p_change_2d(self, y=0.0, path="", format="png", range_lambda_p=range(0, 12)):
        name = f"poisson_distribution_d_function_plot_with_static_y_{y}_lambda_from_{range_lambda_p[0]}_to_{range_lambda_p[-0]}"
        plt.figure()
        plt.grid(True, which='both', linestyle='-', linewidth=1)
        plt.plot(range_lambda_p, [self.d_function(
            y=y,
            lambda_p=lambda_p
        ) for lambda_p in range_lambda_p], color="black")
        plt.title(name)
        plt.xlabel('Y')
        plt.ylabel('X')
        plt.savefig(f'{path}{name}.{format}')


if __name__ == '__main__':
    # nd = NormalDistribution()
    # # nd.plot_count_function(
    # #     std=0.5,
    # #     mean=0.0,
    # #     range_y=float_range(-2.0, 2.0, 0.01)
    # # )
    # # nd.plot_t_function(
    # #     mean=0.5,
    # #     range_y=float_range(-50.0, 50.0, 0.1)
    # # )
    # nd.plot_t_function_mean_change_2d(
    #     y=0,
    #     range_mean=float_range(-5.0, 5.0, 0.1)
    # )
    #
    # # print(nd.t_function(
    # #     y=20,
    # #     m=20
    # # ))
    # #
    # # print(nd.t_function(
    # #     y=0.01,
    # #     m=20
    # # ))
    #
    # ed = ExponentialDistribution()
    #
    # # ed.plot_count_function(
    # #     lambda_p=1.0,
    # #     range_y=float_range(0.0, 5.0, 0.01)
    # # )
    # # ed.plot_t_function(
    # #     lambda_p=1.0,
    # #     range_y=float_range(0.0, 2.0, 0.01)
    # # )
    # # ed.plot_t_function_lambda_p_change_2d(
    # #     y=2.0,
    # #     range_lambda_p=float_range(0.1, 5.0, 0.01)
    # # )
    # ed.plot_d_function_lambda_p_change_2d(
    #     y=0.2,
    #     range_lambda_p=float_range(0.1, 25.0, 0.01)
    # )

    pd = PoissonDistribution()
    pd.plot_count_function(
        lambda_p=0.5,
        range_y=range(0, 15)
    )
    # pd.plot_t_function_lambda_p_change_2d(
    #     y=5.0,
    #     range_lambda_p=float_range(0.01, 40.0, 0.01)
    # )
    # pd.plot_d_function_lambda_p_change_2d(
    #     y=5,
    #     range_lambda_p=float_range(0.1, 30.0, 0.01)
    # )




