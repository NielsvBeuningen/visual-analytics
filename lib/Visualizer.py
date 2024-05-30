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
        
    def dim_reduction(self, customer_row: pd.DataFrame, method: str = "PCA", params: dict = {}) -> None:
        """
        Function to perform dimensionality reduction and plot the reduced data in a scatter plot
        @param method: the method to use for dimensionality reduction
        @param params: the parameters for the dimensionality reduction method
        @return: None
        """
        
        # Add the customer row to the data and labels and store the index
        data = self.data._append(customer_row, ignore_index=True)
        labels = np.append(self.labels, 'Customer')
        
        # Logic to perform the dimensionality reduction method specified
        if method == 'PCA':
            reduced_data = PCA(**params).fit_transform(data)
        elif method == 'tSNE':
            reduced_data = TSNE(**params).fit_transform(self.data)
        elif method == 'MDS':
            reduced_data = MDS(**params).fit_transform(self.data)  
        elif method == "UMAP":
            reduced_data = umap.UMAP(**params).fit(self.data).transform(self.data)
        else:
            raise ValueError('Invalid method')
        
        # Generate the scatter plot
        fig = px.scatter(x=reduced_data[:,0], y=reduced_data[:,1], color=labels)
        
        # Display the scatter plot in the streamlit app
        st.plotly_chart(fig, use_container_width=True)