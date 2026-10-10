# Mental Health Monitoring System

AI-enabled mental health monitoring and clinical consultation system. A Flask web app that classifies fMRI data (116 brain regions) using an SVM model (85-87% accuracy), backed by MySQL.

## Features
- User registration and login
- Upload brain scan data and get a classification result
- Chatbot
- Dashboard

## Tech Stack
Python, Flask, SVM (scikit-learn), MySQL, HTML, CSS, JavaScript

## How to Run
1. Clone the repo: `git clone https://github.com/JeniferBano/mental_health_monitoring.git`
2. Install requirements: `pip install flask scikit-learn mysql-connector-python`
3. Set up your MySQL database and update the connection details in `app.py`
4. Run: `python app.py`
5. Open http://127.0.0.1:5000 in your browser

## Author
Jenifer Bano
GitHub: https://github.com/JeniferBano
