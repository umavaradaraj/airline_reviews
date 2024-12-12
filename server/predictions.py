import numpy as np
import pandas as pd
import pickle

# Load the trained model
file_name = "random_forest_best_model.pkl"
with open(file_name, 'rb') as file:
    loaded_model = pickle.load(file)

# Column names used during training
columns = [
    'Verified', 'Seat Comfort', 'Staff Service', 'Food & Beverages',
    'Inflight Entertainment', 'Value For Money', 'Review_Length',
    'month_flown', 'year_flown', 'review_year', 'reviws_month', 'day',
    'Airline_Air France', 'Airline_All Nippon Airways',
    'Airline_Cathay Pacific Airways', 'Airline_EVA Air', 'Airline_Emirates',
    'Airline_Japan Airlines', 'Airline_Korean Air', 'Airline_Qatar Airways',
    'Airline_Singapore Airlines', 'Airline_Turkish Airlines',
    'Type of Traveller_Business', 'Type of Traveller_Couple Leisure',
    'Type of Traveller_Family Leisure', 'Type of Traveller_Solo Leisure',
    'Class_Business Class', 'Class_Economy Class', 'Class_First Class',
    'Class_Premium Economy'
]


# Example input from UI
user_input = {
    'Verified': 1,
    'Seat Comfort': 5,
    'Staff Service': 5,
    'Food & Beverages': 5,
    'Inflight Entertainment': 4,
    'Value For Money': 4,
    'Review_Length': 556,
    'month_flown': 1,
    'year_flown': 2017,
    'review_year': 2017,
    'reviws_month': 1,
    'day': 31,
    'Airline_Air France': 1,
    'Airline_All Nippon Airways': 0,
    'Airline_Cathay Pacific Airways': 0,
    'Airline_EVA Air': 0,
    'Airline_Emirates': 0,
    'Airline_Japan Airlines': 0,
    'Airline_Korean Air': 0,
    'Airline_Qatar Airways': 0,
    'Airline_Singapore Airlines': 0,
    'Airline_Turkish Airlines': 0,
    'Type of Traveller_Business': 1,
    'Type of Traveller_Couple Leisure': 0,
    'Type of Traveller_Family Leisure': 0,
    'Type of Traveller_Solo Leisure': 0,
    'Class_Business Class': 0,
    'Class_Economy Class': 1,
    'Class_First Class': 0,
    'Class_Premium Economy': 0
}


def make_prediction(user_input):

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure column order matches the training data
    input_df = input_df[columns]

    # Make predictions
    predicted_class = loaded_model.predict(input_df)
    print("Prediction:", predicted_class[0])
    if predicted_class[0] == 1:
        return "Recommended"
    else:
        return "Not Recommended"


print(make_prediction(user_input))
