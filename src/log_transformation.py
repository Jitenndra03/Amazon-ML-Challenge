import pandas as pd
import numpy as np

# --- 1. Define File Paths ---
input_file = '../dataset/submission.csv'
output_file = '../submission_final.csv'

# --- 2. Load the Data ---
try:
    df = pd.read_csv(input_file)
    print(f"Successfully loaded '{input_file}'.")
except FileNotFoundError:
    print(f"Error: Could not find the file '{input_file}'. Please check the file path.")
    exit()

# --- 3. Apply the Inverse Transformation ---
# np.expm1(x) is the inverse of np.log1p(x), calculating e^x - 1.
# This correctly converts your log-transformed prices back to the original scale.
print("Converting 'log_price' back to 'price'...")
df['price'] = np.expm1(df['price'])

# --- 4. Create the Final Submission File ---
# Select only the required columns as per the competition format.
submission_df = df[['sample_id', 'price']]

# Save the final DataFrame to a new CSV file.
# index=False is crucial to avoid writing an extra column.
submission_df.to_csv(output_file, index=False)

print(f"\n✅ Submission file created successfully at: '{output_file}'")
print("\nHere's a preview of the final submission:")
print(submission_df.head())