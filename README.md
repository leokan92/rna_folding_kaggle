# RNA Reactivity Prediction Model

This repository contains Python code to predict the reactivity of RNA sequences to chemical modifiers DMS and 2A3.
The Kaggle competition: [RNA folding competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding)

## Overview

The task involves predicting the reactivity of RNA sequences based on features such as the RNA sequence, experiment type, dataset name, etc. The target variable is an array of floating-point numbers representing the reactivity of each RNA sequence.

## Dependencies

- pandas
- numpy
- scikit-learn

## How to Run

1. Place your `train_data.csv` and `test_data.csv` in a directory.
2. Update the `train_file_path` and `test_file_path` in the Python code to point to your CSV files.
3. Run the Python script to train the model and make predictions.

## Code Explanation

### Functions

- `read_and_preprocess_data(file_path)`: Reads and preprocesses the data from a given CSV file.
- `train_model(train_file_path)`: Trains a linear regression model using the data from `train_file_path`.
- `test_model(model, test_file_path, submission_file_path)`: Tests the model on new data and outputs the results to `submission_file_path`.

### Steps

1. **Data Preprocessing**: The `read_and_preprocess_data` function takes care of reading the data and one-hot encoding the categorical variables.
2. **Model Training**: The `train_model` function trains a Linear Regression model and validates it using a part of the training data.
3. **Model Testing & Submission**: The `test_model` function makes predictions on the test data and saves them to a CSV file.

## Author

Leonardo Kanashiro Felizardo

## License

This project is licensed under the MIT License.
