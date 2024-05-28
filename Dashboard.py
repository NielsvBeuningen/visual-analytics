# Import the required libraries
import streamlit as st

import yaml
from yaml.loader import SafeLoader

import pandas as pd

from lib.Visualizer import Visualizer
from lib.Classifier import Classifier
from lib.Dataloader import Dataloader

# Warnings disabling
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# Try to load the configuration file, if it fails, stop the app
try:
    st.set_page_config(page_title="HELOC Dashboard", layout="wide")

    if "config" not in st.session_state:
        with open('config.yaml') as config_file:
            st.session_state.config = yaml.load(config_file, Loader=SafeLoader)
            
    if "customer_data" not in st.session_state:
        st.session_state.customer_data = {}
        # Use default values for the customer data
        for feature in st.session_state.config["INPUT_VALUES"]:
            st.session_state.customer_data[feature] = st.session_state.config["INPUT_VALUES"][feature]["DEFAULT"]
     
    if "customer_prediction" not in st.session_state:
        st.session_state.customer_prediction = None
       
    if "counterfactuals" not in st.session_state:
        st.session_state.counterfactuals = None
        
    if "output_customer_row" not in st.session_state:
        st.session_state.output_customer_row = None
        
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
    st.session_state.customer_prediction = None
    st.session_state.output_customer_row = None
        
def update_numin() -> None:
    """
    Update the numeric input value when the slider is changed
    @param: None
    @return: None
    """
    feature = st.session_state["feature_select"]
    st.session_state[f'numeric_{feature}'] = st.session_state[f'slider_{feature}']
    st.session_state.customer_data[feature] = st.session_state[f'numeric_{feature}']
    st.session_state.customer_prediction = None
    st.session_state.output_customer_row = None
    
def update_features_vary() -> None:
    """
    Update the features to vary when the select all checkbox is changed
    @param: None
    @return: None
    """
    if st.session_state.select_all:
        st.session_state.features_vary = list(st.session_state.customer_row.columns)
    else:
        st.session_state.features_vary = []
    
def update_select_all() -> None:
    """
    Update the select all checkbox when the features to vary are changed
    @param: None
    @return: None
    """
    if len(st.session_state.features_vary) == len(st.session_state.customer_row.columns):
        st.session_state.select_all = True
    else:
        st.session_state.select_all = False

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


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

if st.session_state.output_customer_row is not None:
    data_exp = st.sidebar.expander("Export Data", expanded=True)
    if st.session_state.counterfactuals is None:
        options = ["Customer Data"]
    else:
        options = ["Customer Data", "Counterfactuals"]
    selected_data = data_exp.multiselect(
        label = "Select data to export", 
        options = options, 
        default = options
        )
    
    filename = data_exp.text_input(
            label="File Name", value="important_data.csv"
            )
    
    # Loop over the selected data to create export df
    export_df = pd.DataFrame()
    for data in selected_data:
        if data == "Customer Data":
            row = st.session_state.output_customer_row
            row.index = ["customer"]
            export_df = export_df._append(row)
        elif data == "Counterfactuals":
            export_df = export_df._append(st.session_state.counterfactuals)
            
    csv = convert_df(export_df)
            
    data_exp.download_button(
        label = "Download Data",
        data = csv,
        file_name = filename,
        mime="text/csv"
    )
        
st.header("Customer Information")
 
# Show the customer data as a dataframe
st.session_state.customer_row = pd.DataFrame(st.session_state.customer_data, index=[0])
customer_features = st.session_state.customer_row.to_numpy()
st.dataframe(st.session_state.customer_row, hide_index=True) 

tab1, tab2 = st.tabs(["Prediction", "Landscape"])

with tab1:
    if st.button("Predict"):
        with st.spinner("Predicting label"):
            st.session_state.customer_prediction = st.session_state.classifier.predict(customer_features)
            st.session_state.output_customer_row = st.session_state.customer_row.copy()
            st.rerun()
            
    if st.session_state.customer_prediction is None:
        st.warning("The current customer data has not been assessed yet. Please click the 'Predict' button to get a prediction.")
        st.stop()

    if st.session_state.customer_prediction["Prediction"] == "Good":
        st.session_state.output_customer_row["RiskPerformance"] = "Good"
        st.success("Loan accepted :smile:")
        st.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][1], 2)}")
    else:
        st.session_state.output_customer_row["RiskPerformance"] = "Bad"
        st.error("Load denied :pensive:")
        st.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][0], 2)}")
        
        st.divider()
        st.subheader("Counterfactuals")
        st.write("Here you can generate counterfactuals for the customer data to find out what changes would be needed for the loan to be accepted.")
        
        cf_exp = st.expander("Counterfactual Settings", expanded=True)
        
        with cf_exp:
            # Just a simple layout for the counterfactual settings
            col1, _, col2, _ = st.columns([1, 1, 3, 6])
            dice_method = col1.selectbox("DiCE Method", ["kdtree", "random", "genetic"])
            n_cfs = col2.slider("Number of Counterfactuals", 1, 10, 1)   
            select_all = st.checkbox("Select all features", value=True, on_change=update_features_vary, key="select_all")     
                        
            features_vary = st.multiselect(
                label="Features to use for counterfactuals", 
                options=st.session_state.customer_row.columns,
                default=list(st.session_state.customer_row.columns),
                on_change=update_select_all,
                key="features_vary"
                )  
        
        if st.button("Generate Counterfactuals"):
            with st.spinner("Generating counterfactuals"):
                st.session_state.counterfactuals = st.session_state.classifier.generate_counterfactuals(
                    show_logs=st.session_state.config["SHOW_LOGS"],
                    method=dice_method,
                    feature_names=list(st.session_state.customer_row.columns), 
                    features_vary=features_vary,
                    customer_data=st.session_state.customer_row,
                    n_cfs=n_cfs)
                st.rerun()
            
        if st.session_state.counterfactuals is None:
            st.warning("The counterfactuals have not been generated yet. Please click the 'Generate Counterfactuals' button to generate them.")
        else:
            st.write("Configurations that would receive a loan approval:")

            # Display the counterfactual result
            st.dataframe(st.session_state.counterfactuals.drop("RiskPerformance", axis=1))
        
with tab2:
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
            st.session_state.visualizer.dim_reduction(method=method, params=params, customer_row=st.session_state.customer_row)

