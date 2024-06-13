import pandas as pd
import numpy as np

class Dataloader:
    def __init__(self, data_path: str):
        """
        Constructor for the Dataloader class. Loads the data and labels into the object and preprocesses the data.
        @param data_path: the path to the data file
        @return: None        
        """
        
        # Load the data
        features = pd.read_csv(data_path)
        
        # Drop na values
        features = features.dropna()
        feature_names = features.columns

        # Determine the label dimension
        labelDimension = "RiskPerformance"
        self.feature_names = feature_names.drop(labelDimension)
        
        # Get the labels
        labels = np.array(features[labelDimension])
        
        # Replace Good with Accepted and Bad with Denied (for better understanding)
        labels = np.where(labels == "Good", "Accepted", "Denied")
        self.labels = labels

        # Remove the labels from the features
        self.original_features = features
        pass