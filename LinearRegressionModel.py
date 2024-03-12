from MachineLearningModel import *


class LinearRegressionModel(MachineLearningModel):
    def __init__(self, start_b0: float = 0, start_b1: float = 0):
        super().__init__()
        self.b0: float = start_b0
        self.b1: float = start_b1

    def learn(self, epochs: int, learning_factor: float = 0.01, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        n, Ex, Ey, Exx, Exy = self.define_learning_data(dataset_with_defined_data_necessity_type)
        loss_history = []
        e = 0
        while e <= epochs:
            epoch_loss = self.loss_function(
                b0=self.b0,
                b1=self.b1,
                data_necessity_type=data_necessity_type
            )

            if e % 10 == 0:
                print(f"Epoch: {e} | Loss: {epoch_loss}")
            loss_history.append({
                "B0": self.b0,
                "B1": self.b1,
                "epoch_loss": epoch_loss
            })

            self.b0 = self.b0 - self.b0_derivative_on_loss(n, Ex, Ey) * learning_factor
            self.b1 = self.b1 - self.b1_derivative_on_loss(n, Exx, Ex, Exy) * learning_factor

            e += 1
        return loss_history

    def b0_derivative_on_loss(self, n: int, Ex: float, Ey: float):
        return 2/n * (self.b1*Ex + n*self.b0 - Ey)

    def b1_derivative_on_loss(self, n: int, Exx: float, Ex: float, Exy: float):
        return 2/n * (self.b1*Exx + self.b0*Ex - Exy)

    def predict(self, x):
        return self.prediction_function(
            x=x,
            b0=self.b0,
            b1=self.b1
        )

    def define_mathematically_ideal_params(self, data_necessity_type: DataNecessityType = DataNecessityType.TRAINING):
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        if dataset_with_defined_data_necessity_type is None:
            raise ValueError("Unable to define dataset with provided data_necessity_type")

        n, Ex, Ey, Exx, Exy = self.define_learning_data(dataset_with_defined_data_necessity_type)

        self.b1 = (n*Exy - Ex*Ey) / (n*Exx - Ex ** 2)
        self.b0 = (Ey - self.b1*Ex) / n

    def loss_function(self, b0: float, b1: float, data_necessity_type: DataNecessityType):
        e = 0
        dataset_with_defined_data_necessity_type = self.dataset.get(data_necessity_type)
        for i in range(dataset_with_defined_data_necessity_type.data_len):
            predicted_y = self.prediction_function(
                x=dataset_with_defined_data_necessity_type.input_data[i],
                b0=b0,
                b1=b1
            )
            e = (predicted_y - dataset_with_defined_data_necessity_type.output_data[i]) ** 2
        return e / dataset_with_defined_data_necessity_type.data_len

    @staticmethod
    def define_learning_data(dataset_with_defined_data_necessity_type: DataSetObject):
        n = dataset_with_defined_data_necessity_type.data_len
        Ex = sum([dataset_with_defined_data_necessity_type.input_data[i] for i in range(n)])
        Ey = sum([dataset_with_defined_data_necessity_type.output_data[i] for i in range(n)])
        Exx = sum([dataset_with_defined_data_necessity_type.input_data[i] ** 2 for i in range(n)])
        Exy = sum([dataset_with_defined_data_necessity_type.input_data[i]
                   * dataset_with_defined_data_necessity_type.output_data[i] for i in range(n)])
        return n, Ex, Ey, Exx, Exy

    @staticmethod
    def prediction_function(x: float, b0: float, b1: float):
        return b0 + b1*x

    def prediction_function_str(self, b0: Optional[float] = None, b1: Optional[float] = None):
        if b0 is None or b1 is None:
            return self._prediction_function_str(b0=self.b0, b1=self.b1)
        else:
            return self._prediction_function_str(b0=b0, b1=b1)

    @staticmethod
    def _prediction_function_str(b0: float, b1: float):
        return f"y = {b0} + {b1}*x"