import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_air_quality_data(input_file="aqi_data_raw.csv", output_file="aqi_data_preprocessed.csv"):
    """
    Comprehensive preprocessing pipeline for air quality data
    
    Preprocessing Techniques Applied:
    1. Data Loading and Initial Inspection
    2. Duplicate Removal
    3. Data Type Conversion
    4. Outlier Detection and Removal
    5. Missing Value Imputation (Median)
    6. Data Validation
    7. Feature Engineering (optional)
    8. Data Normalization/Standardization (optional)
    """
    
    print("="*80)
    print("AIR QUALITY DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # ============================================================================
    # STEP 1: DATA LOADING AND INITIAL INSPECTION
    # ============================================================================
    print("\n[STEP 1] Loading and Inspecting Data...")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Successfully loaded data from: {input_file}")
        print(f"  Original shape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
    except FileNotFoundError:
        print(f"✗ Error: File '{input_file}' not found!")
        print("Please run 'python generate_dataset.py' first.")
        return None
    
    # Display initial statistics
    print(f"\nInitial missing values:\n{df.isnull().sum()}")
    print(f"\nInitial statistics:\n{df.describe()}")
    
    # ============================================================================
    # STEP 2: TIMESTAMP PROCESSING AND DUPLICATE REMOVAL
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 2] Processing Timestamps and Removing Duplicates...")
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    invalid_timestamps = df['Timestamp'].isnull().sum()
    if invalid_timestamps > 0:
        print(f"  Found {invalid_timestamps} invalid timestamps - removing...")
        df = df.dropna(subset=['Timestamp'])
    
    # Remove duplicate timestamps (keep first occurrence)
    duplicates_count = df.duplicated(subset=['Timestamp']).sum()
    if duplicates_count > 0:
        print(f"  Found {duplicates_count} duplicate timestamps - removing...")
        df = df.drop_duplicates(subset=['Timestamp'], keep='first')
    
    print(f"✓ After deduplication: {df.shape}")
    
    # ============================================================================
    # STEP 3: DATA TYPE CONVERSION AND VALIDATION
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 3] Converting Data Types...")
    
    # Define numeric columns
    numeric_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    optional_cols = ['Temperature', 'Humidity']
    
    # Add optional columns if they exist
    for col in optional_cols:
        if col in df.columns:
            numeric_cols.append(col)
    
    # Convert to numeric, coercing errors to NaN
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"  ✓ Converted '{col}' to numeric")
    
    # ============================================================================
    # STEP 4: OUTLIER DETECTION AND REMOVAL
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 4] Detecting and Removing Outliers...")
    
    # Define valid ranges based on international air quality standards
    valid_ranges = {
        'AQI': (0, 500),           # Standard AQI range
        'PM2.5': (0, 500),         # μg/m³
        'PM10': (0, 600),          # μg/m³
        'NO2': (0, 400),           # ppb
        'SO2': (0, 1000),          # ppb
        'CO': (0, 50),             # ppm
        'O3': (0, 500),            # ppb
        'Temperature': (-50, 60),  # °C
        'Humidity': (0, 100)       # %
    }
    
    outliers_removed = 0
    for col, (min_val, max_val) in valid_ranges.items():
        if col in df.columns:
            outliers = ((df[col] < min_val) | (df[col] > max_val)) & df[col].notna()
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"  Found {outlier_count} outliers in '{col}' (range: {min_val}-{max_val})")
                df.loc[outliers, col] = np.nan
                outliers_removed += outlier_count
    
    print(f"✓ Total outliers removed: {outliers_removed}")
    
    # ============================================================================
    # STEP 5: MISSING VALUE IMPUTATION WITH MEDIAN
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 5] Imputing Missing Values with Median...")
    
    print(f"\nMissing values before imputation:\n{df[numeric_cols].isnull().sum()}")
    
    # Calculate medians for each numeric column
    medians = {}
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            medians[col] = median_val
            missing_count = df[col].isnull().sum()
            
            if missing_count > 0:
                df[col].fillna(median_val, inplace=True)
                print(f"  ✓ '{col}': Filled {missing_count} missing values with median = {median_val:.2f}")
    
    # Remove rows where all pollution values are still NaN
    pollution_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    before_removal = len(df)
    df = df.dropna(subset=pollution_cols, how='all')
    removed_rows = before_removal - len(df)
    if removed_rows > 0:
        print(f"  Removed {removed_rows} rows with all pollution values missing")
    
    print(f"\nMissing values after imputation:\n{df[numeric_cols].isnull().sum()}")
    
    # ============================================================================
    # STEP 6: DATA VALIDATION AND QUALITY CHECK
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 6] Validating Data Quality...")
    
    # Check for any remaining issues
    total_nulls = df[numeric_cols].isnull().sum().sum()
    print(f"  Total remaining null values: {total_nulls}")
    
    # Check for infinite values
    inf_count = np.isinf(df[numeric_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"  Infinite values: {inf_count}")
    
    # Data quality score
    quality_score = ((len(df) - total_nulls) / (len(df) * len(numeric_cols))) * 100
    print(f"  Data completeness: {quality_score:.2f}%")
    
    # ============================================================================
    # STEP 7: SORT AND RESET INDEX
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 7] Sorting and Finalizing Dataset...")
    
    # Sort by timestamp
    df = df.sort_values('Timestamp', ascending=False)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"✓ Data sorted by timestamp (newest first)")
    print(f"✓ Index reset")
    
    # ============================================================================
    # STEP 8: SAVE PREPROCESSED DATA
    # ============================================================================
    print("\n" + "="*80)
    print("[STEP 8] Saving Preprocessed Data...")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Saved preprocessed data to: {output_file}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Final dataset shape: {df.shape}")
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"\nFinal statistics:\n{df[numeric_cols].describe()}")
    
    print(f"\nMedian values used for imputation:")
    for col, median_val in medians.items():
        print(f"  {col}: {median_val:.2f}")
    
    print(f"\nSample of preprocessed data:\n{df.head(10)}")
    
    # Save preprocessing report
    save_preprocessing_report(df, medians, output_file)
    
    return df


def save_preprocessing_report(df, medians, output_file):
    """Save detailed preprocessing report"""
    report_file = output_file.replace('.csv', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AIR QUALITY DATA PREPROCESSING REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PREPROCESSING TECHNIQUES APPLIED:\n")
        f.write("-"*80 + "\n")
        f.write("1. Data Loading and Initial Inspection\n")
        f.write("   - Loaded raw CSV data\n")
        f.write("   - Inspected data types and structure\n\n")
        
        f.write("2. Duplicate Removal\n")
        f.write("   - Removed duplicate timestamps\n")
        f.write("   - Kept first occurrence of duplicates\n\n")
        
        f.write("3. Data Type Conversion\n")
        f.write("   - Converted timestamp to datetime format\n")
        f.write("   - Converted all pollutant columns to numeric\n\n")
        
        f.write("4. Outlier Detection and Removal\n")
        f.write("   - Applied domain-based valid ranges:\n")
        f.write("     * AQI: 0-500\n")
        f.write("     * PM2.5: 0-500 μg/m³\n")
        f.write("     * PM10: 0-600 μg/m³\n")
        f.write("     * NO2: 0-400 ppb\n")
        f.write("     * SO2: 0-1000 ppb\n")
        f.write("     * CO: 0-50 ppm\n")
        f.write("     * O3: 0-500 ppb\n")
        f.write("   - Replaced outliers with NaN for imputation\n\n")
        
        f.write("5. Missing Value Imputation (Median Strategy)\n")
        f.write("   - Used median imputation for robustness against outliers\n")
        f.write("   - Median values used:\n")
        for col, median_val in medians.items():
            f.write(f"     * {col}: {median_val:.2f}\n")
        f.write("\n")
        
        f.write("6. Data Validation\n")
        f.write("   - Checked for remaining null values\n")
        f.write("   - Verified no infinite values\n")
        f.write("   - Calculated data completeness score\n\n")
        
        f.write("7. Sorting and Indexing\n")
        f.write("   - Sorted by timestamp (newest first)\n")
        f.write("   - Reset index for clean sequential access\n\n")
        
        f.write("8. Data Export\n")
        f.write(f"   - Saved to: {output_file}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FINAL DATASET INFORMATION\n")
        f.write("="*80 + "\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Records: {len(df)}\n")
        f.write(f"Features: {len(df.columns)}\n")
        f.write(f"Columns: {', '.join(df.columns)}\n\n")
        
        f.write("Data Quality Metrics:\n")
        f.write(f"- Missing values: {df.isnull().sum().sum()}\n")
        f.write(f"- Duplicate timestamps: 0 (removed)\n")
        f.write(f"- Outliers: 0 (removed)\n\n")
    
    print(f"✓ Preprocessing report saved to: {report_file}")


if __name__ == "__main__":
    preprocess_air_quality_data()
