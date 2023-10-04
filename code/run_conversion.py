# Description: It is used to convert the data from csv to parquet.
from code.rna_gnn import DataConverter

# Usage
data_path = 'C:\\Users\\leona\\Desktop\\RNA data'

# For train_data.csv.zip
file_path_train = f'{data_path}/train_data.csv'
parquet_file_path_train = f'{data_path}/train_data.parquet'

# For sample_submission.csv.zip
file_path_sample = f'{data_path}/sample_submission.csv'
parquet_file_path_sample = f'{data_path}/sample_submission.parquet'

# For test_sequences.csv.zip
file_path_test = f'{data_path}/test_sequences.csv'
parquet_file_path_test = f'{data_path}/test_sequences.parquet'

converter = DataConverter()
converter.to_parquet(file_path_train, parquet_file_path_train)
converter.to_parquet(file_path_sample, parquet_file_path_sample)
converter.to_parquet(file_path_test, parquet_file_path_test)