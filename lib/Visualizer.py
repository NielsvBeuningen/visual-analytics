import streamlit as st

import plotly.express as px

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap

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
      
    def color_diff(self, cell_content):
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

    def display_differences(self, df):
        styled_df = df.style.applymap(self.color_diff)
        st.dataframe(styled_df)
        
    def counterfactual_visualization(self, customer_row: pd.DataFrame, cf_df: pd.DataFrame) -> None:
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
                # diff_row.insert(0, "Label", cf_df.loc[index, "Label"])
                diff_row.index = [f"{index} ({cf_df.loc[index, 'Label']})"]
            
            differences = differences._append(diff_row)
        
        # Display the differences in the dashboard
        self.display_differences(differences)
        
    def dim_reduction(self, customer_row: pd.DataFrame, method: str = "PCA", params: dict = {}, counterfactuals: pd.DataFrame = None) -> None:
        """
        Function to perform dimensionality reduction and plot the reduced data in a scatter plot
        @param method: the method to use for dimensionality reduction
        @param params: the parameters for the dimensionality reduction method
        @return: None
        """
        # Add the customer row to the data and labels and store the index
        data = self.data._append(customer_row, ignore_index=True)
        
        labels = np.append(self.labels, 'Customer')
        
        # If counterfactuals are provided, add them to the data and labels
        if counterfactuals is not None:
            data = data.append(counterfactuals, ignore_index=True)
            labels = np.append(labels, ['Alternative']*counterfactuals.shape[0])        
        
        # Logic to perform the dimensionality reduction method specified
        if method == 'PCA':
            reduced_data = PCA(**params).fit_transform(data)
        elif method == 'tSNE':
            reduced_data = TSNE(**params).fit_transform(data)
        elif method == 'MDS':
            reduced_data = MDS(**params).fit_transform(data)  
        elif method == "UMAP":
            reduced_data = umap.UMAP(**params).fit(data).transform(data)
        else:
            raise ValueError('Invalid method')
        
        # Add index to the reduced data linking back to original data, customer row and counterfactuals
        index_column = np.array([f"reference_{i}" for i in range(self.data.shape[0])] + ['customer'])
        
        if counterfactuals is not None:
            index_column = np.append(index_column, [f"alternative_{i}" for i in range(counterfactuals.shape[0])])
            
        reduced_data = np.column_stack((reduced_data, index_column))
        data.index = index_column
        labeled_data = data.copy()
        labeled_data['Label'] = labels
        
        # Create symbols array
        symbols = np.array(['circle']*len(labels))
        symbols[labels == 'Customer'] = 'star'
        symbols[labels == 'Alternative'] = 'square'
        
        BASE_SIZE = 1
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
            symbol=symbols,
            size=sizes,
            size_max=12,
            labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
            hover_name=reduced_data[:,2]
            )
        
        return fig, labeled_data