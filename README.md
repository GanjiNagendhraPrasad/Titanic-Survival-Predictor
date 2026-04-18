<h1 align="center">🚢 Titanic Survival Prediction Web App</h1>

<p align="center">
  <b>Machine Learning | Flask | Deployment</b><br><br>
  🌐 <b>Live App:</b> 
  <a href="https://titanic-survival-predictor-94ur.onrender.com/">
    Titanic Survival Predictor
  </a><br>
  💻 <b>GitHub Repository:</b> 
  <a href="https://github.com/GanjiNagendhraPrasad/Titanic-Survival-Predictor.git">
    View Code
  </a>
</p>

<hr>

<h2>🎯 Project Overview</h2>
<p>
This project is an <b>end-to-end Machine Learning web application</b> that predicts 
whether a passenger survived the Titanic disaster based on user input.
It integrates data preprocessing, model building, and deployment into a single system.
</p>

<hr>

<h2>🧠 Problem Statement</h2>
<p>
The objective is to solve a <b>binary classification problem</b>:
</p>
<ul>
  <li><b>0 → Not Survived</b></li>
  <li><b>1 → Survived</b></li>
</ul>

<hr>

<h2>📊 Dataset</h2>
<p>
The project uses the famous Titanic dataset (Kaggle), which includes:
</p>
<ul>
  <li>Passenger Class (Pclass)</li>
  <li>Gender (Sex)</li>
  <li>Fare</li>
  <li>Siblings/Spouse (SibSp)</li>
  <li>Embarked Location</li>
</ul>

<hr>

<h2>⚙️ Data Preprocessing & Feature Engineering</h2>

<h3>✔ Selected Features</h3>
<ul>
  <li>Pclass_trim</li>
  <li>SibSp_trim</li>
  <li>Fare_trim</li>
  <li>Sex_male</li>
  <li>Embarked_randam_S</li>
</ul>

<h3>✔ Encoding</h3>
<ul>
  <li><b>Sex:</b> Male → 1, Female → 0</li>
  <li><b>Embarked:</b> Southampton (S) → 1, Others → 0</li>
</ul>

<h3>✔ Data Cleaning</h3>
<ul>
  <li>Handled missing values</li>
  <li>Trimmed and transformed numerical features</li>
</ul>

<hr>

<h2>📏 Feature Scaling</h2>
<p>
Used <b>StandardScaler</b> to normalize features:
</p>
<ul>
  <li>Mean = 0</li>
  <li>Standard Deviation = 1</li>
</ul>

<p>
This improves performance of machine learning models by ensuring all features are on the same scale.
</p>

<hr>

<h2>🤖 Model Building</h2>
<ul>
  <li>Trained a classification model (e.g., Logistic Regression)</li>
  <li>Saved model using <code>model.pkl</code></li>
  <li>Saved scaler using <code>standardscalar.pkl</code></li>
</ul>

<hr>

<h2>🌐 Flask Web Application</h2>

<h3>🔹 Backend Workflow</h3>
<ol>
  <li>User inputs data in web form</li>
  <li>Flask receives the data</li>
  <li>Data is converted into DataFrame</li>
  <li>Encoding & preprocessing applied</li>
  <li>Scaling using StandardScaler</li>
  <li>Model predicts output</li>
  <li>Result displayed on UI</li>
</ol>

<hr>

<h2>🎨 Frontend</h2>
<ul>
  <li>Clean and responsive UI</li>
  <li>Gradient background design</li>
  <li>Dropdown inputs for easy selection</li>
  <li>Real-time prediction display</li>
  <li>About Author section included</li>
</ul>

<hr>

<h2>🚀 Deployment</h2>
<p>
The application is deployed on <b>Render</b>:
</p>
<ul>
  <li>Code hosted on GitHub</li>
  <li>Connected repository to Render</li>
  <li>Configured environment and dependencies</li>
  <li>Deployed Flask app for public access</li>
</ul>

<p>
👉 <b>Live Application:</b><br>
<a href="https://titanic-survival-predictor-94ur.onrender.com/">
https://titanic-survival-predictor-94ur.onrender.com/
</a>
</p>

<hr>

<h2>💡 Key Highlights</h2>
<ul>
  <li>✔ End-to-end Machine Learning pipeline</li>
  <li>✔ Feature engineering and preprocessing</li>
  <li>✔ Model + Scaler integration</li>
  <li>✔ Real-time prediction system</li>
  <li>✔ Clean UI/UX design</li>
  <li>✔ Cloud deployment</li>
</ul>

<hr>

<h2>🎯 Future Improvements</h2>
<ul>
  <li>Add prediction probability output</li>
  <li>Use advanced models (Random Forest, XGBoost)</li>
  <li>Add feature importance visualization</li>
  <li>Improve UI with charts</li>
  <li>Add input validation</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>
<p>
<b>Ganji Nagendhra Prasad</b><br>
📧 <a href="mailto:gnagendhraprasad4@gmail.com">gnagendhraprasad4@gmail.com</a>
</p>
