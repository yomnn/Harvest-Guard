from flask import Flask, render_template, request, jsonify
import pymongo
from pymongo import MongoClient 

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['contact_form_db']

# MongoDB connection details
mongodb_host = 'localhost'  # MongoDB server host
mongodb_port = 27017  # MongoDB server port
mongodb_database = 'contact_form_db'  # Name of the database to connect to

# Establish connection to MongoDB server
client = pymongo.MongoClient(mongodb_host, mongodb_port)

# Access the specified database
db = client[mongodb_database]

# Route to render the contact_us.html template
@app.route('/')
def index():
    # Fetch data from MongoDB (example)
    data_from_mongodb = db.contacts.find()
    return render_template('contact_us.html', data=data_from_mongodb)

# Route to handle form submission
@app.route('/submit-form', methods=['POST'])
def submit_form():
    
    # Extract form data from the request
    form_data = request.json
    db.submissions.insert_one(form_data)
    # Insert form data into MongoDB
    # Example: db.your_collection_name.insert_one(form_data)
    
    # Return a response indicating success
    return jsonify({'message': 'Form submitted successfully'}), 200
@app.route('/get-submissions', methods=['GET'])
def get_submissions():
    submissions = list(db.submissions.find())  # Retrieve all submissions from the database
    return jsonify(submissions), 200

if __name__ == '__main__':
    app.run(debug=True)
