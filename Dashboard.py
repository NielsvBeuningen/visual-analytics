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
def convert_df(df: pd.DataFrame) -> bytes:
    """
    Small helper function to convert a dataframe to a csv file
    @param df: the dataframe to convert
    @return: the csv file as bytes
    """
    return df.to_csv().encode("utf-8")


# Generate the sliders and numeric inputs for the features based on a selectbox
st.sidebar.write("Select a feature to update the value")
exp = st.sidebar.expander("Customer Data Updating", expanded=True)
with exp:
    # Create a selectbox to choose the feature to update
    feature = st.selectbox("Feature", list(st.session_state.config["INPUT_VALUES"].keys()), key="feature_select")
    
    # Create a numeric input and a slider for the selected feature
    val = st.number_input('Numeric', value = st.session_state.customer_data[feature], key = f'numeric_{feature}', on_change = update_slider)
    slider_value = st.slider('Slider', min_value = st.session_state.config["INPUT_VALUES"][feature]["MIN"], 
                            value = val, 
                            max_value = st.session_state.config["INPUT_VALUES"][feature]["MAX"],
                            step = 1,
                            key = f'slider_{feature}', on_change= update_numin)

# Create an export section to export the data if the customer data has been assessed
if st.session_state.output_customer_row is not None:
    data_exp = st.sidebar.expander("Export Data", expanded=True)
    
    # Check if counterfactuals have been generated, if not, only allow the customer data to be exported
    if st.session_state.counterfactuals is None or "Error" in st.session_state.counterfactuals:
        options = ["Customer Data"]
    else:
        options = ["Customer Data", "Counterfactuals"]
        
    # Create a multiselect to select the data to export
    selected_data = data_exp.multiselect(
        label = "Select data to export", 
        options = options, 
        default = options
        )
    
    # Create a text input to enter the filename
    filename = data_exp.text_input(
            label="File Name", value="important_data.csv"
            )
    
    # Loop over the selected data to create export dataframe
    export_df = pd.DataFrame()
    for data in selected_data:
        if data == "Customer Data":
            row = st.session_state.output_customer_row
            row.index = ["customer"]
            export_df = export_df._append(row)
        elif data == "Counterfactuals":
            export_df = export_df._append(st.session_state.counterfactuals)
            
    # Convert the dataframe to bytes to be able to download it
    csv = convert_df(export_df)
            
    # Create a download button to download the data
    data_exp.download_button(
        label = "Download Data",
        data = csv,
        file_name = filename,
        mime="text/csv"
    )
        
header = st.container()

# Show the customer data
header.header("Customer Information")
 
# Show the customer data as a dataframe
st.session_state.customer_row = pd.DataFrame(st.session_state.customer_data, index=[0])
customer_features = st.session_state.customer_row.to_numpy()
header.dataframe(st.session_state.customer_row, hide_index=True) 
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

### Custom CSS for the sticky header
st.markdown(
    """
<style>
    div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
        position: sticky;
        top: 2.875rem;
        background-color: white;
        z-index: 999;
    }
    .fixed-header {
        border-bottom: 3px solid #E6E7E9;
    }
</style>
    """,
    unsafe_allow_html=True
)    
    


# Create the tabs for the prediction and the landscape
tab1, tab2 = st.tabs(["Prediction", "Landscape"])

with tab1:
    # This tab is used for assessing the customer data and generating counterfactuals if the loan is denied
    st.subheader("AI Assessment")
    st.write("Click the button below to predict the loan acceptance for the current customer data.")
    
    # Create a button to predict the loan acceptance
    if st.button("Predict"):
        with st.spinner("Predicting label"):
            # Perform the prediction
            st.session_state.customer_prediction = st.session_state.classifier.predict(customer_features)
            st.session_state.output_customer_row = st.session_state.customer_row.copy()
            st.rerun()
            
    # Check if the prediction has been performed
    if st.session_state.customer_prediction is None:
        st.warning("The current customer data has not been assessed yet. Please click the 'Predict' button to get a prediction.")
    else:

        # Display the prediction result and the probability, update the export row based on the prediction
        if st.session_state.customer_prediction["Prediction"] == "Good":
            st.session_state.output_customer_row["RiskPerformance"] = "Good"
            st.success("Loan accepted :smile:")
            st.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][1], 2)}")
        else:
            st.session_state.output_customer_row["RiskPerformance"] = "Bad"
            st.error("Load denied :pensive:")
            st.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][0], 2)}")
            
            # If the loan is denied, show configuration for generating counterfactuals
            st.divider()
            st.subheader("Counterfactuals")
            st.write("Here you can generate counterfactuals for the customer data to find out what changes would be needed for the loan to be accepted.")
            
            # Create an expander for the counterfactual settings
            cf_exp = st.expander("Counterfactual Settings", expanded=True)
            
            with cf_exp:            
                # Allow the user to select the method for generating counterfactuals
                dice_method = st.selectbox("DiCE Method", ["kdtree", "random", "genetic"])
                
                # Allow the user to select the number of counterfactuals to generate
                n_cfs = st.slider("Number of Counterfactuals", 1, st.session_state.config["MAX_CFS"], 1)   
                
                # Checkbox for quick (de)selection of all features
                select_all = st.checkbox("Select all features", value=True, on_change=update_features_vary, key="select_all")     
                            
                # Allow the user to select the features to vary for the counterfactuals generation
                features_vary = st.multiselect(
                    label="Features to use for counterfactuals", 
                    options=st.session_state.customer_row.columns,
                    default=list(st.session_state.customer_row.columns),
                    on_change=update_select_all,
                    key="features_vary"
                    )  
            
            if len(features_vary) == 0:
                st.warning("Please select at least one feature to generate counterfactuals.")
            else:
                # Create a button to generate the counterfactuals
                if st.button("Generate Counterfactuals"):
                    with st.spinner("Generating counterfactuals"):
                        # Generate the counterfactuals using the selected configuration
                        counterfactuals = st.session_state.classifier.generate_counterfactuals(
                            method=dice_method,
                            feature_names=list(st.session_state.customer_row.columns), 
                            features_vary=features_vary,
                            customer_data=st.session_state.customer_row,
                            n_cfs=n_cfs)
                        st.session_state.counterfactuals = counterfactuals[list(st.session_state.customer_row.columns) + ["RiskPerformance"]]
                        st.rerun()
                
            # Check if the counterfactuals have been generated and display them if they are available
            if st.session_state.counterfactuals is None:
                st.warning("The counterfactuals have not been generated yet. Please click the 'Generate Counterfactuals' button to generate them.")
            else:
                if "Error" in st.session_state.counterfactuals:
                    st.error(st.session_state.config["MESSAGES"]["ERRORS"]["CF"])
                    if st.session_state.config["SHOW_LOGS"]: 
                        st.write(st.session_state.counterfactuals[1])
                else:
                    st.write("Configurations that would receive a loan approval together with the difference from the customer data:")

                    # Display the counterfactual result as a dataframe
                    cf_df = st.session_state.counterfactuals.drop("RiskPerformance", axis=1)
                    
                    st.session_state.visualizer.counterfactual_visualization(customer_row=st.session_state.customer_row, counterfactuals=cf_df) 
                    
                    st.info("You can export the data via the **Export Data** section in the sidebar.")
        
        
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
            st.session_state.visualizer.dim_reduction(
                method=method, 
                params=params, 
                customer_row=st.session_state.customer_row,
                counterfactuals=st.session_state.counterfactuals.drop("RiskPerformance", axis=1)
                )

