from flask import Flask, render_template, request
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load the password strength dataset from a CSV file
df = pd.read_csv('passwords.csv')
imputer = SimpleImputer(strategy='most_frequent')
df['password'] = imputer.fit_transform(df[['password']])

# Convert the passwords into a matrix of TF-IDF features
vectorizer = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1, 3))
X_train = vectorizer.fit_transform(df['password'])
y_train = df['strength']

# Train a logistic regression model on the entire dataset
model = LogisticRegression()
model.fit(X_train, y_train)

strength_labels = {
    1: "low",
    2: "medium",
    3: "high"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/password_strength', methods=['POST'])
def password_strength():
    password = request.form['password']

    # Check if the password contains special characters
    has_special_characters = any(c in string.punctuation for c in password)

    # Check if the password contains numbers
    has_numbers = any(c.isdigit() for c in password)

    # Check if the password is at least 16 characters long
    is_at_least_16_characters_long = len(password) >= 16

    # Use the model to predict the strength of the password
    X_new = vectorizer.transform([password])
    strength = model.predict(X_new)[0]

    return render_template('passwordstrength.html', password=password,
                           has_special_characters=has_special_characters,
                           has_numbers=has_numbers,
                           is_at_least_16_characters_long=is_at_least_16_characters_long,
                           strength=strength_labels[strength])

if __name__ == '__main__':
    app.run(debug=True)
