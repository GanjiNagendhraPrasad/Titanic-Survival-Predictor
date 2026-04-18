from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))

# Final columns
final_cols = [
    'Pclass_trim',
    'SibSp_trim',
    'Fare_trim',
    'Sex_male',
    'Embarked_randam_S'
]

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()

        df = pd.DataFrame([{
            'Pclass_trim': float(data['Pclass']),
            'SibSp_trim': float(data['SibSp']),
            'Fare_trim': float(data['Fare'])
        }])

        # Default values
        df['Sex_male'] = 0
        df['Embarked_randam_S'] = 0

        # Encoding
        if data['Sex'] == 'male':
            df['Sex_male'] = 1

        if data['Embarked'] == 'S':
            df['Embarked_randam_S'] = 1

        # Ensure column order
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[final_cols]

        # ✅ APPLY SCALING
        scaled = scaler.transform(df)

        # Prediction
        pred = model.predict(scaled)[0]

        result = "❌ Passenger NOT Survived" if pred == 0 else "✅ Passenger Survived"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)