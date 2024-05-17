# Import the required libraries
import streamlit as st

import yaml
from yaml.loader import SafeLoader

import pandas as pd
import numpy as np

from lib.Visualizer import Visualizer

# Try to load the configuration file, if it fails, stop the app
try:
    if "config" not in st.session_state:
        with open('config.yaml') as config_file:
            st.session_state.config = yaml.load(config_file, Loader=SafeLoader)
            
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
customer_age = st.sidebar.number_input("Customer Age", 18, 100, 25)

def update_slider() -> None:
    """
    Update the slider value when the numeric input is changed
    @param: None
    @return: None
    """
    for feature in st.session_state.config["INPUT_VALUES"]:
        st.session_state[f'slider_{feature}'] = st.session_state[f'numeric_{feature}']
def update_numin() -> None:
    """
    Update the numeric input value when the slider is changed
    @param: None
    @return: None
    """
    for feature in st.session_state.config["INPUT_VALUES"]:
        st.session_state[f'numeric_{feature}'] = st.session_state[f'slider_{feature}']
    
# Loop over all features in the configuration file to generate sliders and numeric inputs
for i, (name, item) in enumerate(st.session_state.config["INPUT_VALUES"].items()):
    
    # CAN BE USED TO CREATE EXPANDERS IN TWO COLUMNS IF NEEDED
    # if i % 2 == 0:
    #     col1, col2 = st.columns(2)
    #     exp = col1.expander(name, expanded=True)
    # else:
    #     exp = col2.expander(name, expanded=True)
    
    # NOW JUST DOING IN IN SIDEBAR
    
    # Generate expander with numeric input and slider for each feature
    exp = st.sidebar.expander(name, expanded=True)
    with exp:
        val = st.number_input('Numeric', value = st.session_state.config["INPUT_VALUES"][name]["DEFAULT"], key = f'numeric_{name}', on_change = update_slider)
        slider_value = st.slider('Slider', min_value = st.session_state.config["INPUT_VALUES"][name]["MIN"], 
                                value = val, 
                                max_value = st.session_state.config["INPUT_VALUES"][name]["MAX"],
                                step = 1,
                                key = f'slider_{name}', on_change= update_numin)
 
# LOAD THE DATA HERE FOR NOW AND THEN PASS TO VISUALIZER
# Load the data
features = pd.read_csv(st.session_state.config["DATA_FILE"])
feature_names = features.columns

# Determine the label dimension
labelDimension = "RiskPerformance"
feature_names = feature_names.drop(labelDimension)
labels = np.array(features[labelDimension])

# Remove the labels from the features
original_features= features.drop(labelDimension, axis = 1)
 
customer_row = pd.DataFrame(columns = original_features.columns)
for feature in original_features.columns:
    customer_row[feature] = [st.session_state[f'numeric_{feature}']]
     
# Create a visualizer object with the data file   
visualizer = Visualizer(original_data=original_features, original_labels=labels, customer_row = customer_row)

# Create an expander for the dimensionality reduction section
with st.expander("Dimensionality Reduction", expanded=True):
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
        visualizer.dim_reduction(method=method, params=params)
