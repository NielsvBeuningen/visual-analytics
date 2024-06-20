# Visual Analytics Setup Guide
## Table of Contents
1. [Setup Repository](#setup-repository)
2. [Setup Streamlit App](#setup-streamlit-app)
3. [Folder and File Descriptions](#folder-and-file-descriptions)

## Setup Repository
Based on whether you have the ZIP file or not, follow the appropriate instructions below. If you have the ZIP file, skip the first step (Clone Repository).

### Clone Repository
First, fork the [repository](https://github.com/NielsvBeuningen/visual-analytics) to your own GitHub account. Then, clone the repository to your local machine.
```bash
git clone https://github.com/YOUR-USERNAME/visual-analytics.git
```

### Install Dependencies
Install Python `3.12` or higher via the [official website](https://www.python.org/downloads/release/python-3122/).

Navigate to the repository and install the required dependencies. (You can also use a virtual environment if you prefer.)
```bash
cd visual-analytics
pip install -r requirements.txt
```

## Setup Streamlit App
### Run Streamlit App
Navigate to the main repo directory and run the Streamlit app.
```bash
cd visual-analytics
streamlit run Dashboard.py
```

If starting for the first time, you will be prompted to enter the Neo4j credentials. Leave empty and press enter to use the default credentials.

## Folder and File Descriptions
- `config.yaml`: Configuration file for the Streamlit app.
- `Dashboard.py`: Main Streamlit app file, contains the layout and functionality of the app.
- `data/`: Contains the datasets and notebooks for data preprocessing and RF model creation.
    - `data/data_preprocessing.ipynb`: Jupyter notebook for data preprocessing.
    - `data/dataset_analysis.ipynb`: Jupyter notebook for dataset analysis.
    - `data/heloc_data_dictionary-2.xlsx`: Data dictionary for the HELOC dataset with metadata.
    - `data/heloc_data_profiling_report_v4.html`: Data profiling report for the HELOC dataset from pandas profiling.
    - `data/heloc_dataset_v4.csv`: Processed HELOC dataset in CSV format.
    - `data/XaiModel.ipynb`: Jupyter notebook for creating the XAI model (mostly provided by the course).
- `lib/`: Contains helper functions for the Streamlit app.
    - `lib/Classifier.py`: Contains the classifier class for the machine learning model and counterfactual generation.
    - `lib/DataLoader.py`: Contains the data loader class for loading the data from the CSV file.
    - `lib/DGrid.py`: Contains the DGrid class for the DGrid visualization.
    - `lib/Visualizer.py`: Contains the visualizer class for all the visualizations.
- `models/`: Contains the trained machine learning models.
    - `models/MainModel_v4.pkl`: Trained Random Forest model for the HELOC dataset.
- `requirements.txt`: Contains the required Python packages for the Streamlit app.