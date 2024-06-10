# Use numpy to convert to arrays
import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Counterfactuals
import dice_ml

# Shap
import shap

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
        
        result = {
            "Prediction": prediction,
            "Prediction Probability": prediction_proba
        }
        
        return result
    
    def generate_counterfactuals(self, method: str, feature_names: list, features_vary: list, customer_data: pd.DataFrame, n_cfs=1):   
             
        # Initialize DiCE model
        d = dice_ml.Data(dataframe=self.data, continuous_features=feature_names, outcome_name='RiskPerformance')
        m = dice_ml.Model(model=self.model, backend="sklearn", model_type='classifier')

        permitted_range = {}

        for feature in features_vary:
            max_value = self.data[feature].max()
            mean_value = self.data[feature].mean()
            max = max_value + mean_value
            permitted_range[feature] = [0, max]
                
        # Generate counterfactuals
        exp = dice_ml.Dice(d, m, method=method)
        
        try:
            counterfactuals = exp.generate_counterfactuals(
                query_instances=customer_data, 
                total_CFs=n_cfs, 
                desired_class="opposite",
                features_to_vary=features_vary,
                permitted_range=permitted_range
                )
        except Exception as e:
            result = ("Error", e)
            return result

        result = counterfactuals.cf_examples_list[0].final_cfs_df
        # Add id column with "counterfactual_i" as index
        result.index = [f"alternative_{i}" for i in range(1, n_cfs+1)]
        
        # Replace Good with Accepted and Bad with Rejected
        result["LoanApplicance"] = np.where(result["RiskPerformance"] == "Good", "Accepted", "Rejected")
        
        result = result.drop(columns=["RiskPerformance"])
        
        return result
    
    def get_shap_values(self, data: np.ndarray):
        """
        Function to get the SHAP values for the data
        """
        
        explainer = shap.TreeExplainer(self.model)
        explanation = explainer(data)

        shap_values = explanation.values
        
        return shap_values
    