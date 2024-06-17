import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import time

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

from lib.DGrid import DGrid

class Visualizer:
    """
    Class to visualize the data, currently only supports dimensionality reduction
    """
    def __init__(self, original_data: pd.DataFrame, original_labels: np.ndarray) -> None:
        """
        Constructor for the Visualizer class. Loads the data and labels into the object
        @param data: the data to visualize, a pandas dataframe
        @param labels: the labels for the data, a numpy array
        @return: None
        """
          
        # Store the data and labels in the object
        self.data = original_data
        self.labels = original_labels
      
    def color_diff(self, cell_content: str) -> str:
        """
        Function to color the differences in the dataframe
        @param cell_content: the content of the cell
        @return: the color of the cell
        """
        # Check if the cell content is a string and contains brackets (indicating a difference)
        if '(' not in cell_content:
            return f'color: black'
        else:
            # Get the value of the cell
            val = float(cell_content.split('(')[-1].split(')')[0])
            
            # Logic to determine the color of the cell
            if val == 0:
                color = 'black'
            elif val < 0:
                color = 'red'
            else:
                color = 'green'
                
            return f'color: {color}'

    def display_differences(self, df: pd.DataFrame) -> None:
        """
        Function to display the differences in a dataframe with color coding
        @param df: the dataframe with the differences
        @return: None
        """
        # Apply the color_diff function to the dataframe for color coding in the dashboard
        styled_df = df.style.applymap(self.color_diff)
        
        # Display the dataframe in the dashboard
        st.dataframe(styled_df)
        
    def counterfactual_visualization(self, customer_row: pd.DataFrame, cf_df: pd.DataFrame) -> None:
        """
        Function to visualize the differences between the customer row and the counterfactuals in a dataframe.
        @param customer_row: the data of the customer
        @param cf_df: the counterfactuals
        @return: None
        """
        
        # Make a copy of the counterfactuals dataframe
        counterfactuals = cf_df.copy()
        
        # If "label" column is present, remove it
        if "Label" in cf_df.columns:
            counterfactuals = counterfactuals.drop(columns=["Label"])
            
        # Use the original data and the counterfactuals to check the difference
        differences = pd.DataFrame()
        for index, row in counterfactuals.iterrows():
            diff_row = pd.DataFrame()
            
            # Get the difference between the customer row and the counterfactual
            difference_values = (row - customer_row)
            
            # Order the values the same way as the original data
            difference_values = difference_values[customer_row.columns]
            row = row[customer_row.columns]
            
            # Flatten the values
            difference_values = difference_values.values.flatten()
            
            # Create a diff overview with the new values and the difference between brackets
            diff_overview = [f"{row.values.flatten()[i]} ({difference_values[i]:+.2f})" for i in range(len(difference_values))]
            
            # Add diff overview as row to diff_row
            diff_row = diff_row._append(pd.Series(diff_overview, index=counterfactuals.columns), ignore_index=True)
            diff_row.index = [index]
            
            if "Label" in cf_df.columns:
                # Add as column or index the label
                diff_row.index = [f"{index} ({cf_df.loc[index, 'Label']})"]
            
            differences = differences._append(diff_row)
        
        # Display the differences in the dashboard with color coding
        self.display_differences(differences)
      
    def difference_visualization(self, customer_row: pd.DataFrame, cf_df: pd.DataFrame, index: int) -> None:
        """
        Function to visualize the difference between the customer row and the counterfactuals in a line plot.
        The counterfactuals are shown as transparent lines, the customer row is shown as a solid line.
        One counterfactual is highlighted in a solid line. The differences are shown as vertical lines.
        @param customer_row: the data of the customer
        @param cf_df: the counterfactuals
        @param index: the index of the counterfactual to highlight
        @return: None
        """
        # Get the data for the counterfactual to highlight        
        cf_filtered = cf_df.loc[[index]]
        
        # Compute differences with the base table for all counterfactuals and the single counterfactual
        differences_all = cf_df.subtract(customer_row.iloc[0])
        differences_single = cf_filtered.subtract(customer_row.iloc[0])

        # Create plotly figure
        fig = go.Figure()

        # Add the customer row as a solid line
        # Add index to hover over the points
        fig.add_trace(go.Scatter(
            x=customer_row.columns,
            y=[0]*len(customer_row.columns),
            mode='markers',
            name='Base',
            marker=dict(size=10, color='grey'),
        ))

        # Create a function to add traces to the plot
        def add_traces(fig: go.Figure, differences: pd.DataFrame, opacity: float) -> go.Figure:
            """
            Function for adding traces to the plot.
            @param fig: the figure to add the traces to
            @param differences: the differences to add to the plot
            @param opacity: the opacity of the traces
            @return: the figure with the traces added
            """
            # Add traces for each alternative
            for idx in differences.index:  
                label = idx.replace('reference_', '')   
                
                # For feature in differences.columns, add a vertical line to show the difference
                for feature in differences.columns:
                    # Color the line based on the difference
                    if differences.loc[idx, feature] > 0:
                        color = 'green'
                    elif differences.loc[idx, feature] < 0:
                        color = 'red'
                    else:
                        color = 'grey'
                        
                    # Add the vertical line to the plot
                    fig.add_shape(
                        type='line',
                        x0=feature,
                        y0=0,
                        x1=feature,
                        y1=differences.loc[idx, feature],
                        line=dict(
                            color=color,
                            width=2
                        ),
                        name=label,
                        opacity=opacity
                    )
                    
                    # Draw circles over points to color them
                    fig.add_trace(go.Scatter(
                        x=[feature],
                        y=[differences.loc[idx, feature]],
                        mode='markers',
                        marker=dict(size=10, color=color),
                        opacity=opacity,
                        name=label
                    ))
                    
            return fig
           
        # First add the differences for all counterfactuals, transparency is set to 0.1  
        fig = add_traces(fig, differences_all, opacity=0.1)
        
        # Then add the differences for the single counterfactual, transparency is set to 1
        fig = add_traces(fig, differences_single, opacity=1)

        # Update layout
        fig.update_layout(
            title='Differences from Customer Data',
            xaxis_title='Feature',
            yaxis_title='Difference',
            yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'),
            template='plotly_white',
            showlegend=False
        )
        
        # Display the plot in the dashboard
        st.plotly_chart(fig)   
        
    def shap_visualization(self, shap_values: np.ndarray, feature_names: list) -> None:
        """
        Function to visualize the SHAP values
        @param shap_values: the SHAP values to visualize
        @param feature_names: the names of the features
        @return: None
        """
        # Create a dataframe with the SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        
        # Create a bar plot with the mean SHAP values
        shap_mean = shap_df.mean().sort_values(ascending=False)
        
        # Create the base for the plot
        fig = px.bar(
            x=shap_mean.values, 
            y=shap_mean.index, 
            orientation='h',
            labels={'x': 'Feature Importance', 'y': 'Feature'}
            )
        
        # Color negative values red and positive values green
        fig.update_traces(marker_color=['red' if x < 0 else 'green' for x in shap_mean.values])        
        
        return fig
            
    def dim_reduction(
        self, column_subset: list,
        customer_row: pd.DataFrame, 
        method: str = "PCA", params: dict = {}, 
        counterfactuals: pd.DataFrame = None,
        glyph_width : float=0.5, 
        glyph_height : float =0.5, 
        delta : float = 20.0
        ) -> None:
        """
        Function to perform dimensionality reduction and plot the reduced data in a scatter plot
        @param method: the method to use for dimensionality reduction
        @param params: the parameters for the dimensionality reduction method
        @return: None
        """
        # Progress bar to show the progress of the dimensionality reduction
        my_bar = st.progress(0, text=f"Performing dimensionality reduction with {method}")
        my_bar.progress(0, text="Loading data")
        
        # Add the customer row to the data and labels and store the index
        data = self.data._append(customer_row, ignore_index=True)
        
        labels = np.append(self.labels, 'Customer')
        
        # If counterfactuals are provided, add them to the data and labels
        if counterfactuals is not None:
            data = data.append(counterfactuals, ignore_index=True)
            labels = np.append(labels, ['Alternative']*counterfactuals.shape[0])        
        
        # Select only the columns that are specified
        filtered_data = data[column_subset]
        
        for percent_complete in range(5):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text="Scaling data")
        
        # Scale the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(filtered_data)
        
        for percent_complete in range(5):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 6, text="Performing dimensionality reduction")
        
        # Logic to perform the dimensionality reduction method specified
        if method == 'tSNE':
            reduced_data = TSNE(**params).fit_transform(data_scaled)
        elif method == "UMAP":
            reduced_data = umap.UMAP(**params).fit(data_scaled).transform(data_scaled)
        else:
            raise ValueError('Invalid method')

        my_bar.progress(50, text="Transforming data for visualization with DGrid")
        
        # Add index to the reduced data linking back to original data, customer row and counterfactuals
        index_column = np.array([f"reference_{i}" for i in range(self.data.shape[0])] + ['customer'])
        
        # If counterfactuals is not None, add the counterfactuals to the index
        if counterfactuals is not None:
            index_column = np.append(index_column, [f"alternative_{i}" for i in range(counterfactuals.shape[0])])
            
        # Use DGrid to transform the data for visualization
        dgrid = DGrid(glyph_width=glyph_width, glyph_height=glyph_height, delta=delta)  # Adjust glyph size and delta as necessary
        reduced_data = dgrid.fit_transform(reduced_data)
            
        my_bar.progress(90, text="Creating visualization")
        
        # Create symbols array    
        reduced_data = np.column_stack((reduced_data, index_column))
        data.index = index_column
        labeled_data = data.copy()
        labeled_data['Label'] = labels
        
        # Create symbols array
        symbols = np.array(['circle']*len(labels))
        symbols[labels == 'Customer'] = 'star'
        symbols[labels == 'Alternative'] = 'square'
        
        # Some constants for the plot visualization
        BASE_SIZE = 0.5
        CUSTOMER = 5
        COUNTERFACTUAL = 5
        
        # Create sizes array
        sizes = np.array([BASE_SIZE]*len(labels))  # default size for original data points
        sizes[labels == 'Customer'] = CUSTOMER    # larger size for customer data
        sizes[labels == 'Alternative'] = COUNTERFACTUAL  # different size for counterfactuals
        
        # Generate the scatter plot
        fig = px.scatter(
            x=reduced_data[:,0], 
            y=reduced_data[:,1], 
            color=labels, 
            color_discrete_sequence=[
                st.session_state.config["VISUALIZATION"]["COLORS"]["ref_bad"],
                st.session_state.config["VISUALIZATION"]["COLORS"]["ref_good"],
                st.session_state.config["VISUALIZATION"]["COLORS"]["customer"],
                st.session_state.config["VISUALIZATION"]["COLORS"]["alternative"]                
            ],
            symbol=symbols,
            size=sizes,
            size_max=12,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            hover_name=reduced_data[:,2]
            )
        
        my_bar.progress(100, text="Creating visualization")
        my_bar.empty()
        return fig, labeled_data