# Nurture_prediction

## Project Description

This project demonstrates a Maternal Health Risk Prediction application. The app uses a machine learning pipeline to classify patients into different risk levels: **High Risk**, **Low Risk**, and **Mid Risk**. It provides a user-friendly Flask-based web application for predictions, data uploads, and retraining of the model.

## Features

1. Prediction Page: Allows users to input patient data and receive risk predictions.

2. Dashboard: Visualizations showcasing insights from the dataset.

3. Data Upload: Users can upload new data to the system for model retraining.

4. Retrain Model: A trigger to retrain the machine learning model based on new data.

5. Flood Simulation: Tests system performance under a high load of prediction requests using Locust.

6. Dockerized Deployment: App is fully containerized for ease of deployment and scaling.

## Video Demo and Live LInk to App

- **Video Demo:**

- **Live App:**

- **Deployed link:** https://nurture-prediction.onrender.com/docs

## Setting Up

1. **Clone the Repository:**

```
git clone https://github.com/k-ganda/nurture_prediction.git
cd maternal_risk_app
```

2. **Set up a virtual environment:**

```
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

`pip install -r requirements.txt`

## Preprocessing

The preprocessing file is `src/preprocessing.py`.

It contains all preprocessing steps; from loading and inspecting, encoding categorical variables, checking for outliers and handling,
scaling

To run:

``python src/preprocessing.py`

## Model Training

The `src/model.py` trains and evaluates the model using the already preprocessed dataset.

To train:

`python src/model.py`

## Prediction

The `src/prediction.py` loads the trained model and makes predictions.

To run predictions:

`python src/prediction.py`

## Notebook

The jupyter notebook contains all the pipeline functions with visualization

To use the notebook ensure you have jupter installed: `pip install jupyter`

Then navigate to the project directory and start jupyter: ``jupyter notebook`

Open the notebook from the jupyter interface.

## Results from Flood Request

To load and run the requests:

``locust --host=https://nurture-prediction.onrender.com`

Open the Locust interface from the link provided in your terminal, configure the number of users and spawn rate, and start the test.

## Contribution Guidelines

Feel free to fork the repositort and submit pull requests for improvements or bug fixes.
