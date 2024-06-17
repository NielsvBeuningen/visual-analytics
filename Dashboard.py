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
        
    if "dim_plot" not in st.session_state:
        st.session_state.dim_plot = None
        st.session_state.dim_plot_data = None

            
except Exception as e:
    st.write(e)
    st.error(st.session_state.config["MESSAGES"]["ERRORS"]["START"])
    st.stop()

# Create the page layout
st.title("Loan AId Dashboard")

# Create reactive functions to update the sliders and numeric inputs
def update_slider() -> None:
    """
    Update the slider value when the numeric input is changed
    @param: None
    @return: None
    """
    feature = st.session_state["feature_select"]
    st.session_state[f'slider_{feature}'] = st.session_state[f'numeric_{feature}']
    st.session_state.customer_data[feature] = st.session_state[f'numeric_{feature}']
    st.session_state.dim_plot = None
        
def update_numin() -> None:
    """
    Update the numeric input value when the slider is changed
    @param: None
    @return: None
    """
    feature = st.session_state["feature_select"]
    st.session_state[f'numeric_{feature}'] = st.session_state[f'slider_{feature}']
    st.session_state.customer_data[feature] = st.session_state[f'numeric_{feature}']
    st.session_state.dim_plot = None
    
# Create reactive functions to update the select boxes and multiselects
def update_features_vary() -> None:
    """
    Update the features to vary when the select all checkbox is changed
    @param: None
    @return: None
    """
    if st.session_state.select_all_predict:
        st.session_state.features_vary = list(st.session_state.customer_row.columns)
    else:
        st.session_state.features_vary = []
        
def update_column_select() -> None:
    """
    Update the features to vary when the select all checkbox is changed
    @param: None
    @return: None
    """
    if st.session_state.select_all_vis:
        st.session_state.column_select = list(st.session_state.customer_row.columns)
    else:
        st.session_state.column_select = []
    
def update_select_all_predict() -> None:
    """
    Update the select all checkbox when the features to vary are changed
    @param: None
    @return: None
    """
    if len(st.session_state.features_vary) == len(st.session_state.customer_row.columns):
        st.session_state.select_all_predict = True
    else:
        st.session_state.select_all_predict = False
        
def update_select_all_vis() -> None:
    """
    Update the select all checkbox when the features to vary are changed
    @param: None
    @return: None
    """
    if len(st.session_state.column_select) == len(st.session_state.customer_row.columns):
        st.session_state.select_all_vis = True
    else:
        st.session_state.select_all_vis = False

# Create reactive functions to reset the dimensionality reduction plot
def reset_dim_plot() -> None:
    """
    Reset the dim plot when the method for dimensionality reduction is changed
    @param: None
    @return: None
    """
    st.session_state.dim_plot = None
    st.session_state.dim_plot_data = None

# Create a function to convert a dataframe to a csv file (cached to avoid recomputation)
@st.cache_data
def convert_df(df: pd.DataFrame) -> bytes:
    """
    Small helper function to convert a dataframe to a csv file
    @param df: the dataframe to convert
    @return: the csv file as bytes
    """
    return df.to_csv().encode("utf-8")

st.sidebar.title("Dashboard Options")

# Generate the sliders and numeric inputs for the features based on a selectbox
with st.sidebar.popover(":grey_question:"):
    st.subheader("Loan AId Sidebar")
    st.write(
        """
        This sidebar consists of three components.
        1. **Customer Data Updating**: Here you can select a customer attribute from the
        dropdown menu and update the value using the numeric input or the slider.
        The features can be seen in the **Customer Information** section on the right 
        side of the page.\n
        2. **Export Data**: This section allows you to export the customer data and the 
        counterfactuals generated for the customer data. You can select the data to 
        export and enter the filename for the exported data (Only shown when a 
        prediction for the current customer has been made).\n
        3. **Customer Data View Orientation**: This section allows you to switch 
        between the row and column view of the customer data. You can also lock the
        customer data view to the top of the page, making it easier to compare the
        customer data with the counterfactuals and reference points.
        """
    )
exp = st.sidebar.expander("**Customer Data Updating**", expanded=True)
with exp:
    # Create a selectbox to choose the feature to update
    feature = st.selectbox("Feature", list(st.session_state.config["INPUT_VALUES"].keys()), key="feature_select")
    
    if f'numeric_{feature}' not in st.session_state:
        st.session_state[f'numeric_{feature}'] = st.session_state.customer_data[feature]
    if f'slider_{feature}' not in st.session_state:
        st.session_state[f'slider_{feature}'] = st.session_state.customer_data[feature]
    
    # Create a numeric input and a slider for the selected feature
    val = st.number_input('Numeric', key = f'numeric_{feature}', on_change = update_slider)
    slider_value = st.slider('Slider', min_value = st.session_state.config["INPUT_VALUES"][feature]["MIN"], 
                            max_value = st.session_state.config["INPUT_VALUES"][feature]["MAX"],
                            step = 1,
                            key = f'slider_{feature}', on_change= update_numin)

# Create an export section to export the data if the customer data has been assessed
if st.session_state.output_customer_row is not None:
    data_exp = st.sidebar.expander("**Export Data**", expanded=True)
    
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

customer_view_exp = st.sidebar.expander("**Customer Data View**", expanded=True)

toggle_lock = customer_view_exp.toggle("Lock View", value=True)

# Create a toggle to switch between row and column view
toggle_display = customer_view_exp.selectbox("Customer Data View Orientation", ["Rows", "Columns"])

if toggle_display == "Columns":
    nr_columns = customer_view_exp.slider("Number of columns", 1, 4, 4)
    columns = header.columns(nr_columns)
    column_dfs = []
    for i, col in enumerate(columns):
        column_dfs.append(st.session_state.customer_row.iloc[:, i::nr_columns].transpose())
        column_dfs[i].columns = ["Value"]
        columns[i].dataframe(column_dfs[i])    
else:
    header.dataframe(st.session_state.customer_row, hide_index=True) 

if toggle_lock:
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
    col1, _, col2 = st.columns([5, 5, 1])
    # This tab is used for assessing the customer data and generating counterfactuals if the loan is denied
    col1.title("AI Assessment")
    with col2.popover(":grey_question:"):
        st.subheader("AI Assessment")
        st.write(
            """
            Here you can assess the current customer data and predict the loan acceptance.
            The prediction is done with a trained machine learning model which returns the
            prediction and the prediction probability.\n
            On the right side, you can see the
            SHAP analysis for the prediction, showing the importances of all customer attributes
            for the prediction. If the loan is denied, you can generate counterfactuals to 
            find out what changes would be needed for the loan to be accepted.
            """
        )
        
    col1, col2 = st.columns(2)
    
    col1.subheader("Prediction")
        
    subcol1, subcol2 = col1.columns([1, 4])
    subcol2.write("Click the button to predict the loan acceptance for the current customer data.")
    
    # Create a button to predict the loan acceptance
    if subcol1.button("Predict"):
        with st.spinner("Predicting label"):
            customer_features = st.session_state.customer_row.to_numpy()
            
            # Perform the prediction
            prediction = st.session_state.classifier.predict(customer_features)
            
            shap_values = st.session_state.classifier.get_shap_values(customer_features)
            
            if prediction["Prediction"] == "Good":
                st.session_state.customer_prediction = {
                    "Prediction": "Accepted",
                    "Prediction Probability": prediction["Prediction Probability"],
                    "SHAP Values": shap_values[0][:, 1].reshape(1, -1)
                }
            else:
                st.session_state.customer_prediction = {
                    "Prediction": "Denied",
                    "Prediction Probability": prediction["Prediction Probability"],
                    "SHAP Values": shap_values[0][:, 0].reshape(1, -1)
                }
                            
            st.session_state.output_customer_row = st.session_state.customer_row.copy()
            st.session_state.counterfactuals = None
            st.rerun()
    
    # Check if the prediction has been performed
    if st.session_state.customer_prediction is None:
        st.warning("The current customer data has not been assessed yet. Please click the 'Predict' button to get a prediction.")
    else:        
        # Display the prediction result and the probability, update the export row based on the prediction
        if st.session_state.customer_prediction["Prediction"] == "Accepted":
            st.session_state.output_customer_row["LoanApplicance"] = "Accepted"
            col1.success("Loan accepted :smile:")
            col1.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][1], 2)}")
            
            col2.subheader("SHAP Analysis")
            fig = st.session_state.visualizer.shap_visualization(
                shap_values=st.session_state.customer_prediction["SHAP Values"], 
                feature_names=st.session_state.customer_row.columns
            )
            # Display the plot in the dashboard
            col2.plotly_chart(fig)
        else:
            st.session_state.output_customer_row["LoanApplicance"] = "Denied"
            col1.error("Load denied :pensive:")
            col1.info(f"Probability: {round(st.session_state.customer_prediction["Prediction Probability"][0], 2)}")
            
            col2.subheader("SHAP Analysis")
            fig = st.session_state.visualizer.shap_visualization(
                shap_values=st.session_state.customer_prediction["SHAP Values"], 
                feature_names=st.session_state.customer_row.columns
            )
            # Display the plot in the dashboard
            col2.plotly_chart(fig)
            
            # If the loan is denied, show configuration for generating counterfactuals
            st.divider()
            col1, _, col2 = st.columns([5, 5, 1])
            col1.subheader("Alternative Generation")
            with col2.popover(":grey_question:"):
                st.subheader("Alternative Generation")
                st.write(
                    """
                    Here you can generate alternatives for the customer data to 
                    find out what changes would be needed for the loan to be accepted.\n
                    These suggestions are generated based on the current customer data 
                    and the selected features to vary. Using a method from the DiCE library,
                    you can generate a number of **counterfactuals** that could act as
                    suggestions for the customer data to be accepted.
                    """
                )
            
            # Create an expander for the counterfactual settings
            cf_exp = st.expander("**Counterfactual Settings**", expanded=True)
            
            with cf_exp:            
                # Allow the user to select the method for generating counterfactuals
                dice_method = st.selectbox("DiCE Method", ["kdtree", "random", "genetic"])
                
                # Allow the user to select the number of counterfactuals to generate
                n_cfs = st.slider("Number of Alternatives", 1, st.session_state.config["MAX_CFS"], 1)   
                
                # Checkbox for quick (de)selection of all features
                select_all = st.checkbox(
                    "Select all features", value=True, 
                    on_change=update_features_vary, 
                    key="select_all_predict")     
                            
                if "features_vary" not in st.session_state:
                    st.session_state.features_vary = list(st.session_state.customer_row.columns)    
                        
                # Allow the user to select the features to vary for the counterfactuals generation
                features_vary = st.multiselect(
                    label="Features to use for counterfactuals", 
                    options=st.session_state.customer_row.columns,
                    on_change=update_select_all_predict,
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
                        if type(counterfactuals) == tuple:
                            st.error("No counterfactuals could be generated for the current customer data. Please try again with different settings.")
                        else:
                            st.session_state.counterfactuals = counterfactuals[list(st.session_state.customer_row.columns) + ["LoanApplicance"]]
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
                    # Display the counterfactuals in the dashboard
                    st.write("Configurations that would receive a loan approval together with the difference from the customer data:")

                    # Create a radio button to select the method for displaying the counterfactuals
                    method = st.radio("Select the method to display the counterfactuals", ["Plot", "Table"])
                    
                    # Display the counterfactual result as a dataframe
                    cf_df = st.session_state.counterfactuals.drop("LoanApplicance", axis=1)
                    
                    # Display the counterfactuals in the selected method
                    if method == "Plot":
                        row = st.selectbox("Select a counterfactual", cf_df.index)
                        with st.spinner("Updating Figure..."):
                            st.session_state.visualizer.difference_visualization(customer_row=st.session_state.customer_row, cf_df=cf_df, index = row)
                    else:
                        st.session_state.visualizer.counterfactual_visualization(customer_row=st.session_state.customer_row, cf_df=cf_df) 
                    
                    st.info("You can inspect the counterfactuals in the **Landscape** tab, or export the data via the **Export Data** section in the sidebar.")
        
        
with tab2:
    col1, _, col2 = st.columns([5, 5, 1])
    col1.title("Landscape")
    with col2.popover(":grey_question:"):
        st.subheader("Landscape")
        st.write(
            """
            In this tab, you can visualize the customer data, the counterfactuals, 
            and the reference points in one large figure.\n
            The scatter plot is generated using dimensionality reduction, for which you can
            select the method and columns.\n
            You can select points in the plot to compare these with the customer data.
            """
        )
    
    if st.session_state.counterfactuals is None:
        st.info("The landscape view shows the **customer** and **reference points**")
    else:
        st.info("The landscape view shows the **customer**, **counterfactuals**, and **reference points**")
    
    column_exp = st.expander("**Column Selection**", expanded=True)
    column_exp.write("Select the columns to use for the dimensionality reduction")
    
    # Checkbox for quick (de)selection of all features
    select_all = column_exp.checkbox(
        "Select all columns", value=True, 
        on_change=update_column_select, 
        key="select_all_vis"
        )
    
    # Input for selecting which columns to use for the dimensionality reduction
    selected_columns = column_exp.multiselect(
        "Select columns for dimensionality reduction", 
        st.session_state.customer_row.columns.to_list(), 
        st.session_state.customer_row.columns.to_list(),
        key="column_select",
        on_change=update_select_all_vis
        )
    
    # Create a selectbox to choose the method for dimensionality reduction
    method = st.selectbox("Method", ["UMAP", "tSNE"], on_change=reset_dim_plot)
    
    col1, col2 = st.columns(2)
    
    # Create the settings for the dimensionality reduction
    dgrid_exp = col1.expander("**DGrid Settings**", expanded=False)
                
    # Make the sliders for the DGrid settings
    glyph_width = dgrid_exp.slider(
        label="Glyph Width", min_value=0.1, 
        max_value=st.session_state.config["VISUALIZATION"]["DGRID"]["MAX_GLYPH_WIDTH"], 
        value = 0.2)
    glyph_height = dgrid_exp.slider(
        label="Glyph Height", min_value=0.1,
        max_value=st.session_state.config["VISUALIZATION"]["DGRID"]["MAX_GLYPH_HEIGHT"],
        value = 0.2)
    delta = dgrid_exp.slider(
        label="Delta", min_value=0.1, 
        max_value=st.session_state.config["VISUALIZATION"]["DGRID"]["MAX_DELTA"], 
        value = 50.0)
    
    # Create the settings for the dimensionality reduction
    method_exp = col2.expander(f"**{method} Settings**", expanded=False)
        
    # tSNE method        
    if method == "tSNE":
        perplexity = method_exp.slider("Perplexity", 5, 100, 30)
        n_iter = method_exp.slider("Number of Iterations", 250, 1000, 250)
        params = {
            "n_components": 2, 
            "random_state": 42,
            "perplexity": perplexity, 
            "n_jobs": -1,
            "n_iter": n_iter
            }
    # UMAP method
    elif method == "UMAP":
        n_neighbors = method_exp.slider("Number of Neighbors", 2, 100, 15)
        min_dist = method_exp.slider("Minimum Distance", 0.01, 0.99, 0.1)
        params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": 2
            }    
    
    # React to the button press and perform the dimensionality reduction
    if st.button("Apply"):    
        if st.session_state.counterfactuals is None:
            cf_df = None
        else:
            cf_df = st.session_state.counterfactuals.drop("LoanApplicance", axis=1)
        

        # with st.spinner(f"Performing dimensionality reduction with {method}"):
        st.session_state.dim_plot, st.session_state.dim_plot_data = st.session_state.visualizer.dim_reduction(
            column_subset=selected_columns,
            method=method, 
            params=params, 
            customer_row=st.session_state.customer_row,
            counterfactuals=cf_df,
            glyph_width=glyph_width,
            glyph_height=glyph_height,
            delta=delta
            )
           
    # Check if the dimensionality reduction has been performed and display the scatter plot
    if st.session_state.dim_plot is None:
        st.info("Please click the **Apply** button to perform the dimensionality reduction.")
    else:
        # Display the scatter plot in the streamlit app
        event = st.plotly_chart(st.session_state.dim_plot, key="dim_red_plot", use_container_width=True, on_select="rerun")
        
        # Check if points have been selected in the plot, if so, display the selected data
        selected_points = event.selection["points"]
        selected_ids = [point["hovertext"] for point in selected_points]
        if len(selected_ids) == 0:
            st.info("**Select** points in the plot to compare with the **customer data**.")
        else:
            method = st.selectbox("Method", ["Plot", "Table"])
            
            # Get the selected data based on the selected ids
            selected_data = st.session_state.dim_plot_data[st.session_state.dim_plot_data.index.isin(selected_ids)]
            
            # Reorder the columns to match the customer data but keep index
            selected_data = selected_data[st.session_state.customer_row.columns.to_list() + ["Label"]]
                
            if method == "Plot":
                # Provide input for selecting the data point to highlight
                selected_data.index = [f"{index} ({selected_data.loc[index, 'Label']})" for index in selected_data.index]
                highlight_index = st.selectbox("Select a data point", selected_data.index)
                
                with st.spinner("Loading data..."):
                    st.session_state.visualizer.difference_visualization(customer_row=st.session_state.customer_row, cf_df=selected_data, index=highlight_index)
            else:
                with st.spinner("Loading data..."):
                    # Display the selected data in a table for comparison
                    st.session_state.visualizer.counterfactual_visualization(customer_row=st.session_state.customer_row, cf_df=selected_data) 
