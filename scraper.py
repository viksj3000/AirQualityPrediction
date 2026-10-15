import pandas as pd

data = {
    'mood': ['tired', 'energetic', 'tired', 'energetic', 'stressed', 'relaxed', 'stressed', 'relaxed', 'energetic', 'relaxed'],
    'evening_weather': ['pleasant', 'pleasant', 'rainy', 'rainy', 'pleasant', 'pleasant', 'humid', 'humid', 'very_hot', 'very_hot'],
    'will_buy_street_food': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no']
}

df = pd.DataFrame(data)
print(df)