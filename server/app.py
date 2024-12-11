from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# List to store submitted data
data_storage = []

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

            # Predict sentiment
            review = form_data["Review"]
            review_vectorized = vectorizer.transform([review])
            prediction = model.predict(review_vectorized)[0]  # 1 for positive, 0 for negative
            sentiment = "Positive" if prediction == 1 else "Negative"

            # Store data
            form_data["Sentiment"] = sentiment
            data_storage.append(form_data)

            # Return sentiment with an icon
            return render_template(
                'result.html',
                sentiment=sentiment,
                review=review,
            )

        except ValueError as e:
            return f"<h1>Error: Invalid input detected! {str(e)}</h1>", 400

if __name__ == '__main__':
    app.run(debug=True)
