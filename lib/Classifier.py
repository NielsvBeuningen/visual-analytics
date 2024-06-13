# Import the required libraries
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
            # Load the model from the pickle file and store it in the object
            self.model = pickle.load(open(file_path, 'rb'))
            
        except Exception as e:
            st.error(st.session_state.config["MESSAGES"]["ERRORS"]["MODEL"])
        
        # Load the data into the object
        self.data = data

        
    def predict(self, data: np.ndarray) -> tuple[str, float]:
        """
        Function to perform prediction on the data.
        @param data: the data to predict on.
        @return: the prediction and the prediction probability.
        """
        
        # Get the prediction and the prediction probability from the model
        prediction = self.model.predict(data)[0]
        prediction_proba = self.model.predict_proba(data)[0]
        
        # Return the prediction and the prediction probability in a dictionary
        result = {
            "Prediction": prediction,
            "Prediction Probability": prediction_proba
        }
        
        return result
    
    def generate_counterfactuals(
        self, method: str, 
        feature_names: list, features_vary: list, 
        customer_data: pd.DataFrame, n_cfs: int = 1) -> pd.DataFrame:   
        """
        Function to generate counterfactuals for the data using the DiCE library.
        @param method: the method to use for generating counterfactuals
        @param feature_names: the names of the features in the data
        @param features_vary: the features to vary in the counterfactuals
        @param customer_data: the data of the customer
        @param n_cfs: the number of counterfactuals to generate
        @return: the counterfactuals
        """
        # Initialize DiCE model
        d = dice_ml.Data(dataframe=self.data, continuous_features=feature_names, outcome_name='RiskPerformance')
        m = dice_ml.Model(model=self.model, backend="sklearn", model_type='classifier')

        # Define permitted range for each feature to vary
        permitted_range = {}
        for feature in features_vary:
            max_value = self.data[feature].max()
            mean_value = self.data[feature].mean()
            max = max_value + mean_value
            permitted_range[feature] = [0, max]
                
        # Generate counterfactuals
        exp = dice_ml.Dice(d, m, method=method)
        
        try:
            # Generate counterfactuals
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

        # Get the counterfactuals from the result
        result = counterfactuals.cf_examples_list[0].final_cfs_df
        
        # Add id column with "alternative_i" as index
        result.index = [f"alternative_{i}" for i in range(1, n_cfs+1)]
        
        # Replace Good with Accepted and Bad with Rejected with LoanApplicance column (for better understanding)
        result["LoanApplicance"] = np.where(result["RiskPerformance"] == "Good", "Accepted", "Rejected")
        
        # Remove RiskPerformance column
        result = result.drop(columns=["RiskPerformance"])
        
        return result
    
    def get_shap_values(self, data: np.ndarray) -> np.ndarray:
        """
        Function to get the SHAP values for the data using the SHAP library.
        @param data: the data to get the SHAP values for.
        @return: the SHAP values.
        """
        
        # Initialize the explainer from the SHAP library
        explainer = shap.TreeExplainer(self.model)
        explanation = explainer(data)

        # Get the SHAP values
        shap_values = explanation.values
        
        return shap_values
    