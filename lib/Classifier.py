# Use numpy to convert to arrays
import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Counterfactuals
import dice_ml
from dice_ml.utils import helpers

class Classifier:
    def __init__(self, file_path: str, data: pd.DataFrame) -> None:
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
        
        self.data = data

        
    def predict(self, data: np.ndarray) -> tuple[str, float]:
        """
        Function to perform prediction on the data
        """
        
        prediction = self.model.predict(data)[0]
        prediction_proba = self.model.predict_proba(data)[0]
        
        return prediction, prediction_proba
    
    def generate_counterfactuals(self, feature_names: list, customer_data: pd.DataFrame, n_cfs=1) -> None:   
        
             
        # Initialize DiCE model
        d = dice_ml.Data(dataframe=self.data, continuous_features=feature_names, outcome_name='RiskPerformance')
        m = dice_ml.Model(model=self.model, backend="sklearn", model_type='classifier')

        # FOR TESTING
        # customer_data = self.data.tail(1).drop("RiskPerformance", axis=1)
        
        # Generate counterfactuals
        exp = dice_ml.Dice(d, m)
        
        try:
            counterfactuals = exp.generate_counterfactuals(
                query_instances=customer_data, 
                total_CFs=n_cfs, 
                desired_class="opposite"
                )
        except Exception as e:
            st.error(st.session_state.config["MESSAGES"]["ERRORS"]["CF"])
            st.write(e)
            return None

        st.write("Counterfactuals")

        # Display the counterfactual result
        st.dataframe(counterfactuals.cf_examples_list[0].final_cfs_df)