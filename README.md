# Visual Analytics Setup Guide
## Table of Contents
1. [Setup Repository](#setup-repository)
2. [Setup Streamlit App](#setup-streamlit-app)

## Setup Repository
### Clone Repository
First, fork the repository to your own GitHub account. Then, clone the repository to your local machine.
```bash
git clone https://github.com/YOUR-USERNAME/visual-analytics.git
```

### Install Dependencies
Install Python 3.12 or higher.

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