import wfdb
import pandas as pd
import os

# Path to the record (without extension)
record_path =  r"data\ptb-xl-a-large-publicly-availabl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\records100\00000\00674_lr"

# Read the record
record = wfdb.rdrecord(record_path)

# Convert to pandas DataFrame
df = pd.DataFrame(record.p_signal, columns=record.sig_name)

# Create folder for CSVs
csv_folder = "csv"
os.makedirs(csv_folder, exist_ok=True)

# Extract patient ID and record ID from path
patient_id = os.path.basename(os.path.dirname(record_path))
record_id = os.path.basename(record_path)

# Construct CSV filename
csv_filename = f"{patient_id}_{record_id}.csv"
csv_path = os.path.join(csv_folder, csv_filename)

# Save CSV
df.to_csv(csv_path, index=False)

print(f"CSV saved at: {csv_path}")
