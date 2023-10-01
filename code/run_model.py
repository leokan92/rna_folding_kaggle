from rna_model import RNAPrediction

if __name__ == "__main__":
    # Data path
    data_path = 'C:\\Users\\leona\\Desktop\\RNA data'

    # Initialize the RNAPrediction class
    rna_prediction = RNAPrediction(train_parquet_file=f'{data_path}/train_data.parquet',
                                   test_parquet_file=f'{data_path}/test_sequences.parquet',
                                   edge_distance=5)
    
    # Train the model
    rna_prediction.train_model(n_epochs=10)
    
    # Load the saved model
    rna_prediction.load_model("best_model.pth")

    # Run inference to generate predictions
    rna_prediction.inference()
    
    # Save the submission file
    submission_file_path = f"{data_path}/sample_submission.csv"
    rna_prediction.save_submission(submission_file_path)
