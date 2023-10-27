# Flight Fare Price Prediction Project

## Overview:

This project aims to predict flight fare prices based on several input features, such as airline, source, destination, total stops, and duration and date. The project utilizes machine learning techniques such as preprocessing, transformation and random forest regressor to create a predictive model that can estimate flight fares accurately.

## Folder Structure:

Here's an overview of the project's folder structure:

```
Flight-Fare-Price-Project/
├── artifacts/
│   (Include trained models or other important artifacts)
│
├── notebooks/
│   ├── notebook.ipynb (Your Jupyter Notebook for the project)
│   ├── data (Data files or datasets)
│
├── logs/ (Log files or logs directory)
│
├── src/
│   ├── logger.py (Logging utilities)
│   ├── exception.py (Custom exception handling)
│   ├── utils.py (Utility functions)
│   ├── __init__.py
│
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py (Data loading functions)
│   │   ├── data_transformation.py (Data preprocessing and feature engineering)
│   │   ├── model_training.py (Machine learning model training)
│
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py (Fare prediction pipeline)
│   │   ├── training_pipeline.py (Model training pipeline)
│
│   
├── visuals/
│   ├── _image.jpg (Images, plots, or visualization files)
│
├── requirements.txt (List of project dependencies)
├── setup.py (Setup script for the project)
├── app.py (Main application script)
├── .gitignore (Specify files or directories to ignore in version control)

```

## Project Workflow:

1. **Data Ingestion and Preprocessing:**

   - Data is ingested using functions in `data_ingestion.py`.
   - Data preprocessing and feature engineering are performed in `data_transformation.py`.

2. **Model Training:**

   - Machine learning models are trained using the data in `model_training.py`.
   - You can customize the models and hyperparameters as needed.

3. **Prediction Pipeline:**

   - The `prediction_pipeline.py` script handles fare predictions based on user input.

4. **Notebook:**

   - The `notebook.ipynb` contains an interactive Jupyter Notebook that provides insights into the project.

## Technologies used:

1. Python, AWS Ec2 instance.
2. Libraries used : Streamlit, Pandas, Numpy, Sklearn, Seaborn, etc.
3. IDE: VS-code

## Execution:

To run the project:
1. Install the required dependencies listed in `requirements.txt`.
2. Navigate to the `notebooks/` directory and open `notebook.ipynb` to explore and experiment with the project interactively.
3. Use the scripts in the `src/` directory to execute various components of the project.
4. Customize and extend the project as needed for your specific use case.

### Testing:

To run the web app follow this link: https://flight-price-prediction-jtpxo7e5ubbnkpbehuoee9.streamlit.app/

## Author:

[Yash Keshari]
Socials: https://www.linkedin.com/in/yash907/
