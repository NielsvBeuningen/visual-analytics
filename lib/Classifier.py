# Use numpy to convert to arrays
import numpy as np
import pickle
import streamlit as st

class Classifier:
    def __init__(self, file_path: str) -> None:
        """
        Constructor for the Classifier class. Loads the data and labels into the object
        @param data: the data to classify, a pandas dataframe
        @param labels: the labels for the data, a numpy array
        @return: None
        """
        try:
            self.model = pickle.load(open(file_path, 'rb'))
            
        except Exception as e:
            st.error(st.session_state.config["MESSAGES"]["ERRORS"]["MODEL"])
        
    def predict(self, data: np.ndarray) -> tuple[str, float]:
        """
        Function to perform prediction on the data
        """
        
        prediction = self.model.predict(data)[0]
        prediction_proba = self.model.predict_proba(data)[0]
        
        return prediction, prediction_proba