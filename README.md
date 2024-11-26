Iris Flower Classification Project
This repository contains a machine learning project for classifying Iris flowers into their respective species: Setosa, Versicolor, and Virginica. 
The project leverages the classic Iris dataset, a popular dataset for introductory classification tasks. 
The goal is to train a model that accurately predicts flower species based on their sepal and petal measurements.

Project Structure
IRIS.csv: The dataset used for this project.
classification.py: File containing the complete code, analysis, and visualization steps.
README.md: Project overview and instructions for running the code.

Tools & Libraries
The following Python libraries were used:
pandas: Data manipulation and analysis
numpy: Numerical computations
matplotlib & seaborn: Data visualization
scikit-learn: Machine learning algorithms and evaluation
PCA: Dimensionality reduction for visualization

Steps Implemented
Data Exploration:
Pairplots and heatmaps to understand feature relationships and correlations.
Visualization of class separability based on measurements.

Data Preprocessing:
Label encoding of target variable (species).
Splitting data into training (80%) and testing (20%) sets.

Model Training & Validation:
Trained a Random Forest Classifier.
Performed 5-fold cross-validation to ensure robust evaluation.

Model Evaluation:
Evaluated using classification reports, confusion matrix, and accuracy score.
Achieved 100% accuracy on the test set.

Dimensionality Reduction:
Applied PCA for 2D visualization of the dataset.

Prediction on New Data:
Demonstrated model predictions on new samples.
Key Findings
The Random Forest Classifier provided excellent performance, achieving 100% test accuracy and robust cross-validation scores.
Visualizations revealed clear separability among the species based on petal and sepal measurements.
PCA effectively reduced dimensionality, confirming data clustering for the three species.

How to Run
Clone the repository:
git clone https://github.com/your_username/Iris-Flower-Classification.git  
Install the required dependencies:
pip install -r requirements.txt  
Run the code in classification in your preferred IDE.

Results
Cross-Validation Accuracy: 97%
Test Set Accuracy: 100%
Confusion Matrix: Perfect classification with no misclassifications.

Acknowledgment
This project is part of my CodSoft Data Science Internship. Special thanks to CodSoft for the opportunity to work on this classic yet insightful dataset.
