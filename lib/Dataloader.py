import pandas as pd
import numpy as np

class Dataloader:
    def __init__(self, data_path: str):
        features = pd.read_csv(data_path)
        feature_names = features.columns

        # Determine the label dimension
        labelDimension = "RiskPerformance"
        feature_names = feature_names.drop(labelDimension)
        
        self.labels = np.array(features[labelDimension])

        # Remove the labels from the features
        self.original_features= features.drop(labelDimension, axis = 1)
        pass
        