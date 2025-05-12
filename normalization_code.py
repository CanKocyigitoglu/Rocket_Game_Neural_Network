import pandas as pd

file_path = 'ce889_dataCollection.csv'
data = pd.read_csv(file_path, header=None)

normalized_data = (data - data.min()) / (data.max() - data.min())

output_path = 'normalized_data.csv'
normalized_data.to_csv(output_path, index=False, header=False)