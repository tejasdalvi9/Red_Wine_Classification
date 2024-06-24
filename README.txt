Wine Quality Prediction Using PCA and Random Forest
Overview
This project aims to predict the quality of red wine using a Random Forest Classifier. The dataset used is the "Wine Quality" dataset, which includes various physicochemical properties of wine. The project involves dimensionality reduction using Principal Component Analysis (PCA) and classification using a Random Forest model.

Dataset
Source: UCI Machine Learning Repository
Attributes:
fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
quality (target variable)
Project Steps
Data Loading and Exploration:

Load the dataset and explore its structure and summary statistics.
Data Preprocessing:

Standardize the dataset to have zero mean and unit variance.
Handle missing values and perform necessary cleaning.
Dimensionality Reduction using PCA:

Apply PCA to reduce the number of features while retaining 95% of the variance.
Visualize the explained variance to determine the optimal number of components.
Model Training and Evaluation:

Split the data into training and testing sets.
Train a Random Forest Classifier on the training data.
Evaluate the model using confusion matrix and accuracy score.
Installation
To run this project, you need to have Python and the following libraries installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
You can install the required libraries using pip:

Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
Clone the repository:

Copy code
git clone https://github.com/tejasdalvi9/Red_Wine_Classification.git
Navigate to the project directory:


Copy code
cd Red_Wine_Classification
Run the Jupyter notebook or Python script to preprocess the data, apply PCA, and train the model:


Copy code
jupyter notebook Wine_Quality_Prediction.ipynb
or


Copy code
python wine_quality_prediction.py

Results
Accuracy: The Random Forest model achieved an accuracy of 72.19% on the test set.

Confusion Matrix:
[[  0   1   0   1   0   0]
 [  0   0   7   5   0   0]
 [  0   0 109  19   1   0]
 [  0   1  33 108   5   0]
 [  0   0   1  11  14   0]
 [  0   0   0   2   2   0]]

Visualizations
Correlation Heatmap:

Shows the correlation between different features.
Explained Variance by PCA Components:

Plots the cumulative explained variance to determine the optimal number of PCA components.
Future Enhancements

Model Tuning:
Tune the hyperparameters of the Random Forest model to improve accuracy.
Experiment with other classifiers like Gradient Boosting, SVM, or Neural Networks.
Feature Engineering:

Explore additional features or transformations that could improve model performance.
Cross-Validation:

Implement k-fold cross-validation to get a more robust estimate of the model's performance.
Deployment:

Consider deploying the model as a web application using Flask or Django.

Additional Visualizations:
Create more visualizations to understand the distribution and relationships of features.

Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

Contact
If you have any questions or suggestions, feel free to reach out to me at tejasdalvi9130@gmail.com