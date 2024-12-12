from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained models
sentiment_model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

with open('random_forest_best_model.pkl', 'rb') as file:
    recommendation_model = pickle.load(file)

# List to store submitted data
data_storage = []

# Column names used during training for the recommendation model
recommendation_columns = [
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

MONTH_MAPPING = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = {
                "Airline": request.form.get("airline"),
                "Type of Traveller": request.form.get("traveller_type"),
                "Month Flown": request.form.get("month_flown"),
                "Class": request.form.get("class"),
                "Seat Comfort": int(request.form.get("seat_comfort", 0)),
                "Staff Service": int(request.form.get("staff_service", 0)),
                "Food & Beverages": int(request.form.get("food_beverage", 0)),
                "Inflight Entertainment": int(request.form.get("inflight_entretainment", 0)),
                "Value For Money": int(request.form.get("value_for_money", 0)),
                "Overall Rating": int(request.form.get("overall_rating", 0)),
                "Review": request.form.get("review", "").strip()
            }

            # Validate required fields
            if not all([form_data["Airline"], form_data["Type of Traveller"], form_data["Month Flown"], form_data["Class"]]):
                return "<h1>Error: Missing required fields!</h1>", 400
            
            form_data["Month Flown"] = MONTH_MAPPING.get(form_data["Month Flown"], 0)  # Default to 0 if invalid

            # Predict sentiment
            review = form_data["Review"]
            review_vectorized = vectorizer.transform([review])
            sentiment_prediction = sentiment_model.predict(review_vectorized)[0]  # 1 for positive, 0 for negative
            sentiment = "Positive" if sentiment_prediction == 1 else "Negative"

            # Prepare data for recommendation prediction
            user_input = {
                'Verified': 1,  # Placeholder; adjust based on user input if available
                'Seat Comfort': form_data["Seat Comfort"],
                'Staff Service': form_data["Staff Service"],
                'Food & Beverages': form_data["Food & Beverages"],
                'Inflight Entertainment': form_data["Inflight Entertainment"],
                'Value For Money': form_data["Value For Money"],
                'Review_Length': len(review),
                'month_flown': form_data["Month Flown"], # Map to numeric
                'year_flown': 2023,  # Placeholder
                'review_year': 2023,  # Placeholder
                'reviws_month': form_data["Month Flown"],
                'day': 1,  # Placeholder
                **{f"Airline_{airline}": 1 if airline == form_data["Airline"] else 0 for airline in [
                    'Air France', 'All Nippon Airways', 'Cathay Pacific Airways', 'EVA Air', 'Emirates',
                    'Japan Airlines', 'Korean Air', 'Qatar Airways', 'Singapore Airlines', 'Turkish Airlines'
                ]},
                **{f"Type of Traveller_{traveller}": 1 if traveller == form_data["Type of Traveller"] else 0 for traveller in [
                    'Business', 'Couple Leisure', 'Family Leisure', 'Solo Leisure'
                ]},
                **{f"Class_{class_type}": 1 if class_type == form_data["Class"] else 0 for class_type in [
                    'Business Class', 'Economy Class', 'First Class', 'Premium Economy'
                ]}
            }

            # Convert user input to DataFrame and reorder columns
            input_df = pd.DataFrame([user_input])[recommendation_columns]

            # Predict recommendation
            recommendation_prediction = recommendation_model.predict(input_df)[0]  # 1 for recommended, 0 for not recommended
            recommendation = "Recommended" if recommendation_prediction == 1 else "Not Recommended"

            # Store data
            form_data["Sentiment"] = sentiment
            form_data["Recommendation"] = recommendation
            data_storage.append(form_data)

            # Return sentiment and recommendation
            return render_template(
                'result.html',
                sentiment=sentiment,
                recommendation=recommendation,
                review=review,
            )

        except ValueError as e:
            return f"<h1>Error: Invalid input detected! {str(e)}</h1>", 400

if __name__ == '__main__':
    app.run(debug=True)
