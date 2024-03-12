from PredictionType import *
from DataNecessityType import *


class DataType:
    def __init__(self, prediction_type: PredictionType, data_necessity_type: DataNecessityType):
        self.prediction_type = prediction_type
        self.data_necessity_type = data_necessity_type
