import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_comprehensive_dataset(num_samples=100):
    """
    Generate comprehensive air quality dataset with all pollutants
    """
    print("Generating comprehensive air quality dataset...")
    
    data_list = []
    base_time = datetime.now()
    
    # Multiple locations in Hyderabad
    locations = [
        "Hyderabad-Central", "Hyderabad-Industrial", "Hyderabad-Residential", 
        "Hyderabad-Highway", "Hyderabad-Park", "Hyderabad-Commercial",
        "Hyderabad-Downtown", "Hyderabad-Suburb"
    ]
    
    for i in range(num_samples):
        # Simulate different pollution levels and conditions
        pollution_factor = random.uniform(0.5, 2.5)
        is_missing = random.random() < 0.15  # 15% chance of missing values
        is_outlier = random.random() < 0.05   # 5% chance of outliers
        
        # Generate data with some missing values and outliers
        data = {
            "Timestamp": (base_time - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "Location": random.choice(locations),
            "AQI": int(random.uniform(50, 400) * pollution_factor) if not is_missing else None,
            "PM2.5": round(random.uniform(15, 200) * pollution_factor, 2) if random.random() > 0.1 else None,
            "PM10": round(random.uniform(30, 350) * pollution_factor, 2) if random.random() > 0.1 else None,
            "NO2": round(random.uniform(10, 150), 2) if random.random() > 0.08 else None,
            "SO2": round(random.uniform(5, 80), 2) if random.random() > 0.12 else None,
            "CO": round(random.uniform(0.5, 15), 2) if random.random() > 0.1 else None,
            "O3": round(random.uniform(20, 180), 2) if random.random() > 0.09 else None,
            "Temperature": round(random.uniform(20, 38), 1) if random.random() > 0.05 else None,
            "Humidity": round(random.uniform(30, 85), 1) if random.random() > 0.05 else None
        }
        
        # Add some outliers
        if is_outlier:
            outlier_col = random.choice(['PM2.5', 'PM10', 'AQI'])
            if outlier_col == 'AQI':
                data['AQI'] = random.choice([600, -10, 550])
            elif outlier_col == 'PM2.5':
                data['PM2.5'] = random.choice([600, -5, 550])
            elif outlier_col == 'PM10':
                data['PM10'] = random.choice([700, -8, 650])
        
        # Add some duplicate timestamps (to test duplicate removal)
        if random.random() < 0.08:  # 8% duplicates
            data["Timestamp"] = (base_time - timedelta(hours=max(0, i-1))).strftime("%Y-%m-%d %H:%M:%S")
        
        data_list.append(data)
    
    df = pd.DataFrame(data_list)
    return df

if __name__ == "__main__":
    # Generate dataset
    df = generate_comprehensive_dataset(150)
    
    # Save to CSV
    df.to_csv("aqi_data_raw.csv", index=False)
    print(f"\n✓ Generated {len(df)} records")
    print(f"✓ Saved to: aqi_data_raw.csv")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 10 rows:\n{df.head(10)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
