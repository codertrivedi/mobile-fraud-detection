from flask import Flask, request, render_template
import tensorflow as tf
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model, tokenizer, and scaler
model = tf.keras.models.load_model('my_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define all possible transaction types for one-hot encoding
transaction_types = ['CASH_IN', 'CASH_OUT', 'TRANSFER', 'DEBIT', 'PAYMENT']

# Define the expected column order as per training
expected_columns = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest',
    'tp_CASH_IN', 'tp_CASH_OUT', 'tp_DEBIT', 
    'tp_PAYMENT', 'tp_TRANSFER'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'type': request.form['type'],
                'amount': float(request.form['amount']),
                'nameDest': request.form['nameDest'],
                'oldbalanceOrg': float(request.form['oldbalanceOrg']),
                'newbalanceOrig': float(request.form['newbalanceOrig']),
                'oldbalanceDest': float(request.form['oldbalanceDest']),
                'newbalanceDest': float(request.form['newbalanceDest'])
            }
            
            # Preprocess the input data
            df = pd.DataFrame([data])

            # Process nameDest using the tokenizer and convert to input
            customers_input = tokenizer.texts_to_sequences(df['nameDest'])
            customers_input = tf.keras.preprocessing.sequence.pad_sequences(customers_input, maxlen=1)
            
            # Drop nameDest column since it's now processed
            df = df.drop('nameDest', axis=1)

            # One-hot encode the type column
            df = onehot_encode(df, column='type', prefix='tp')
            
            # Ensure all possible transaction types are present
            for tp in transaction_types:
                if f'tp_{tp}' not in df.columns:
                    df[f'tp_{tp}'] = 0

            # Reorder columns to match the training order
            df = df[expected_columns]

            # Scale the input features
            df = scaler.transform(df)

            # Model prediction
            prediction = model.predict([df, customers_input])[0][0]
            prediction_result = "Fraud" if prediction >= 0.5 else "Not Fraud"

            return render_template('index.html', prediction_result=prediction_result)
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    return render_template('index.html')

def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

if __name__ == '__main__':
    app.run(debug=True)
