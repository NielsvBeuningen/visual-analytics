# Import the required libraries
import streamlit as st

import yaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np

from lib.Visualizer import Visualizer
from lib.Classifier import Classifier
from lib.Dataloader import Dataloader

# Try to load the configuration file, if it fails, stop the app
try:
    if "config" not in st.session_state:
        with open('config.yaml') as config_file:
            st.session_state.config = yaml.load(config_file, Loader=SafeLoader)
            
    if "customer_data" not in st.session_state:
        st.session_state.customer_data = {}
        # Use default values for the customer data
        for feature in st.session_state.config["INPUT_VALUES"]:
            st.session_state.customer_data[feature] = st.session_state.config["INPUT_VALUES"][feature]["DEFAULT"]
        
    if "dataloader" not in st.session_state:
        st.session_state.dataloader = Dataloader(
            data_path = st.session_state.config["DATA_FILE"]
            )
            
    if "classifier" not in st.session_state:
        st.session_state.classifier = Classifier(
            file_path = st.session_state.config["MODEL_FILE"], 
            data=st.session_state.dataloader.original_features
            )
        
    if "visualizer" not in st.session_state:
        st.session_state.visualizer = Visualizer(
            original_data = st.session_state.dataloader.original_features.drop("RiskPerformance", axis = 1), 
            original_labels = st.session_state.dataloader.labels
            )

            
except Exception as e:
    st.write(e)
    st.error(st.session_state.config["MESSAGES"]["ERRORS"]["START"])
    st.stop()

# Create the page layout
st.set_page_config(page_title="HELOC Dashboard", layout="wide")
st.title("HELOC Dashboard")
st.sidebar.title("Customer Information")

# CUSTOMER NAME NEEDED?
# customer_name = st.sidebar.text_input("Customer Name", "John Doe")
# customer_age = st.sidebar.number_input("Customer Age", 18, 100, 25)

def update_slider() -> None:
    """
    Update the slider value when the numeric input is changed
    @param: None
    @return: None
    """
    feature = st.session_state["feature_select"]
    st.session_state[f'slider_{feature}'] = st.session_state[f'numeric_{feature}']
    st.session_state.customer_data[feature] = st.session_state[f'numeric_{feature}']
        
def update_numin() -> None:
    """
    Update the numeric input value when the slider is changed
    @param: None
    @return: None
    """
    feature = st.session_state["feature_select"]
    st.session_state[f'numeric_{feature}'] = st.session_state[f'slider_{feature}']
    st.session_state.customer_data[feature] = st.session_state[f'numeric_{feature}']

# Generate the sliders and numeric inputs for the features based on a selectbox
st.sidebar.write("Select a feature to update the value")
exp = st.sidebar.expander("Feature updates", expanded=True)
with exp:
    feature = st.selectbox("Feature", list(st.session_state.config["INPUT_VALUES"].keys()), key="feature_select")
    val = st.number_input('Numeric', value = st.session_state.customer_data[feature], key = f'numeric_{feature}', on_change = update_slider)
    slider_value = st.slider('Slider', min_value = st.session_state.config["INPUT_VALUES"][feature]["MIN"], 
                            value = val, 
                            max_value = st.session_state.config["INPUT_VALUES"][feature]["MAX"],
                            step = 1,
                            key = f'slider_{feature}', on_change= update_numin)

    
# Loop over all features in the configuration file to generate sliders and numeric inputs
# for i, (name, item) in enumerate(st.session_state.config["INPUT_VALUES"].items()):
    
    # CAN BE USED TO CREATE EXPANDERS IN TWO COLUMNS IF NEEDED
    # if i % 2 == 0:
    #     col1, col2 = st.columns(2)
    #     exp = col1.expander(name, expanded=True)
    # else:
    #     exp = col2.expander(name, expanded=True)
    
    # NOW JUST DOING IN IN SIDEBAR
    
    # Generate expander with numeric input and slider for each feature
    # exp = st.sidebar.expander(name, expanded=True)
    # with exp:
    #     val = st.number_input('Numeric', value = st.session_state.config["INPUT_VALUES"][name]["DEFAULT"], key = f'numeric_{name}', on_change = update_slider)
    #     slider_value = st.slider('Slider', min_value = st.session_state.config["INPUT_VALUES"][name]["MIN"], 
    #                             value = val, 
    #                             max_value = st.session_state.config["INPUT_VALUES"][name]["MAX"],
    #                             step = 1,
    #                             key = f'slider_{name}', on_change= update_numin)
 
# LOAD THE DATA HERE FOR NOW AND THEN PASS TO VISUALIZER
# Load the data

 
# Show the customer data as a dataframe
st.session_state.customer_row = pd.DataFrame(st.session_state.customer_data, index=[0])
customer_features = st.session_state.customer_row.to_numpy()
st.dataframe(st.session_state.customer_row, hide_index=True) 

if st.button("Predict"):
    with st.spinner("Predicting label"):
        prediction = st.session_state.classifier.predict(customer_features)

        if prediction[0] == "Good":
            st.success("Loan accepted :)")
            st.info(f"Probability: {prediction[1][1]}")
            
            # Add "RiskPerformance" to customer row
            st.session_state.customer_row["RiskPerformance"] = "Good"
        else:
            st.error("Load denied :(")
            st.info(f"Probability: {prediction[1][0]}")
        
if st.button("Generate Counterfactuals"):
    with st.spinner("Generating counterfactuals"):
        feature_names = ["ExternalRiskEstimate"]
        st.session_state.classifier.generate_counterfactuals(feature_names=feature_names, customer_data=st.session_state.customer_row)
    
# Create a visualizer object with the data file   

# Create an expander for the dimensionality reduction section
with st.expander("Landscape", expanded=True):
    st.write("Use the dropdown to select the method for dimensionality reduction")
    
    # Create a selectbox to choose the method for dimensionality reduction
    method = st.selectbox("Method", ["PCA", "tSNE"])
    
    # PCA method
    if method == "PCA":
        params = {
            "n_components": 2,
            "random_state": 42
            }
        
    # tSNE method        
    elif method == "tSNE":
        perplexity = st.slider("Perplexity", 5, 100, 30)
        n_iter = st.slider("Number of Iterations", 250, 1000, 250)
        params = {
            "n_components": 2, 
            "random_state": 42,
            "perplexity": perplexity, 
            "n_jobs": -1,
            "n_iter": n_iter
            }
                
    # Create a placeholder for the button to apply the dimensionality reduction (just for ui styling)
    start_btn = st.empty()
    
# React to the button press and perform the dimensionality reduction
if start_btn.button("Apply"):    
    with st.spinner(f"Performing dimensionality reduction with {method}"):
        st.session_state.visualizer.dim_reduction(method=method, params=params, customer_row=customer_row)
