ml-service 🤖

Mentora's ML microservice — a Flask REST API serving a scikit-learn career prediction model, deployed independently from the main Next.js app.

Part of the Mentora platform.

📌 What It Does
This service exposes a REST endpoint that the Mentora frontend calls to generate ML-based career insights for users — such as salary predictions, role-fit scores, or skill gap analysis — based on a trained scikit-learn model.
Separating this into its own service means:

The model can be retrained and redeployed without touching the main app
Python ML ecosystem (scikit-learn, pandas, numpy) stays isolated from the Node.js frontend
The service can be scaled independently if inference becomes a bottleneck


🛠️ Tech Stack
LibraryVersionPurposeFlask3.0.0REST API frameworkFlask-CORS4.0.0Cross-origin requests from Next.jsscikit-learn1.4.0ML model (training & inference)pandas2.1.4Data preprocessingnumpy1.26.3Numerical operationsjoblib1.3.2Model serialization (.pkl)gunicorn21.2.0Production WSGI server

📁 Project Structure
ml-service/
├── app.py              # Flask app — API routes & inference logic
├── train_model.py      # Model training script
├── utils.py            # Preprocessing & helper functions
├── model/              # Saved model file(s) (.pkl)
├── data/               # Training dataset(s)
├── requirements.txt    # Python dependencies
└── runtime.txt         # Python version for deployment (e.g. python-3.11.x)

🚀 Getting Started
Prerequisites

Python 3.10+
pip

1. Clone the repo
bashgit clone https://github.com/ishanikapoor197/ml-service.git
cd ml-service
2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Train the model (if no pre-trained model exists)
bashpython train_model.py
This will read from data/, train the model, and save the serialized output to model/.
5. Run the development server
bashpython app.py
The service will be available at http://localhost:5000.

📡 API Reference
POST /predict
Returns a career prediction or insight score for the given user profile.
Request body:
json{
  "industry": "Software Engineering",
  "experience_years": 3,
  "skills": ["Python", "React", "SQL"],
  "education": "Bachelor's"
}
Response:
json{
  "predicted_salary": 95000,
  "role_fit_score": 0.82,
  "top_skill_gaps": ["System Design", "Cloud Infrastructure"]
}

⚠️ The exact fields depend on your trained model. Update this section to match your actual request/response schema.

GET /health
Simple health check — returns 200 OK if the service is running.
json{ "status": "ok" }

🧠 Model Details

Algorithm: (e.g., Random Forest / Gradient Boosting — update with your actual model)
Training data: Located in data/ — tabular career dataset
Serialization: Model saved with joblib to model/model.pkl
Retraining: Run python train_model.py to retrain from scratch


🌍 Deployment
The service is deployed on Render (or Heroku — as specified by runtime.txt).
Deploy to Render

Create a new Web Service on render.com
Connect this GitHub repo
Set:

Build command: pip install -r requirements.txt
Start command: gunicorn app:app


Add any required environment variables in Render's dashboard
Deploy!

Environment Variables
env# Add any secrets your app.py uses, for example:
FLASK_ENV=production
# Any API keys or config values

🔗 Integration with Mentora
In the main Mentora Next.js app, set this environment variable:
envML_SERVICE_URL=https://your-ml-service.onrender.com
The Next.js API routes will call ML_SERVICE_URL/predict when generating career insights for users.

📝 Notes

Make sure __pycache__/ is in your .gitignore — it shouldn't be committed
The model/ directory contains the trained .pkl file — commit it if it's small enough, otherwise use Git LFS or regenerate on deploy via train_model.py


👥 Authors
Built with ❤️‍🔥 by Ishan and Ishani — part of the Mentora project.
