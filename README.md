# RNA Reactivity Prediction Model

## Overview

This repository contains a specialized implementation for predicting RNA reactivity using Graph Neural Networks (GNN) built on PyTorch and PyTorch Geometric. This project was inspired by the Kaggle competition [RNA folding competition](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding).

## Dependencies

- PyTorch
- PyTorch Geometric
- Pandas
- NumPy
- scikit-learn
- tqdm
- polars

## Directory Structure

```
code/
|-- rna_model.py
|-- run_model.py
```

## Installation

To install the required packages, run:

```bash
pip install torch torch-geometric pandas numpy scikit-learn tqdm polars
```

## How to Run

1. Place your `train_data.parquet` and `test_sequences.parquet` in a directory.
2. Update the `data_path` in `run_model.py` to point to your Parquet files.
3. Run `run_model.py` to train the model, perform inference, and generate the submission file.

```bash
python run_model.py
```

## Code Explanation

### Classes and Functions

- `SimpleGraphDataset`: A custom PyTorch Geometric Dataset class for loading and transforming the RNA sequence data.
- `RNAPrediction`: Main class containing methods for training, inference, and submission file creation.
- `train_model`: Trains the EdgeCNN model and performs validation.
- `inference`: Performs inference on the test data.
- `save_model` and `load_model`: Methods for saving and loading the trained model.

### Steps

1. **Data Preprocessing**: `SimpleGraphDataset` reads the Parquet files and transforms the sequence data into graph form. It also handles one-hot encoding of the RNA sequence and constructs edge indices for the graph.
2. **Model Training**: `train_model` method is responsible for training the EdgeCNN model using the Adam optimizer. It also performs validation and saves the best model.
3. **Inference & Submission**: `inference` method generates the predictions for the test set. `save_submission` method creates the submission file in the required format.

## Author

Leonardo Kanashiro Felizardo

## License

This project is licensed under the MIT License.
