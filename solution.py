import pandas as pd
import numpy as np

# --- Extra Step: Dummy Dataset Generate Karna (Kyunke aapke paas file nahi ha) ---
print("[*] Generating synthetic HousePricePrediction dataset for testing...")
np.random.seed(42)

# Khali dabbe (missing values) dalne ke liye array bana rahe hain
def introduce_nan(arr, percentage=0.1):
    n = len(arr)
    indices = np.random.choice(n, int(n * percentage), replace=False)
    arr_with_nan = arr.astype(object)
    arr_with_nan[indices] = np.nan
    return arr_with_nan

# 1000 gharon ka fake data create karna
fake_id = np.arange(1, 1001)
fake_lotsize = introduce_nan(np.random.randint(1500, 10000, size=1000))
fake_rooms = introduce_nan(np.random.randint(2, 6, size=1000))
fake_type = introduce_nan(np.random.choice(['Apartment', 'Villa', 'Townhouse'], size=1000, p=[0.5, 0.3, 0.2]))
fake_price = np.random.randint(50000, 500000, size=1000)

data = {
    'Id': fake_id,
    'LotSize': fake_lotsize,
    'Rooms': fake_rooms,
    'Type': fake_type,
    'Price': fake_price
}

# DataFrame banana aur as a fallback file save karna
df_generated = pd.DataFrame(data)
df_generated.to_csv('HousePricePrediction.csv', index=False)
print("[*] 'HousePricePrediction.csv' created successfully in your folder!\n")


# =====================================================================
# Task 1: Dataset Loading
df = pd.read_csv('HousePricePrediction.csv')

print("--- Task 1: Dataset Loading (First 10 Rows) ---")
print(df.head(10))

# Task 2: Data Exploration
print("\n--- Task 2: Data Exploration ---")
print("Shape of the dataset:", df.shape)
print("\nData types of columns:")
print(df.dtypes)
print("\nSummary statistics of numerical features:")
print(df.describe())

# Task 3: Data Cleaning
print("\n--- Task 3: Data Cleaning ---")
print("Missing values per column before cleaning:")
print(df.isnull().sum())

# Handling missing values
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Task 4: Feature Selection
print("\n--- Task 4: Feature Selection ---")
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)
    print("'Id' column removed successfully.")

# Task 5: Data Preprocessing
print("\n--- Task 5: Data Preprocessing (Encoding) ---")
df_processed = pd.get_dummies(df)
print("\nProcessed Dataset Preview (First 5 Rows):")
print(df_processed.head())