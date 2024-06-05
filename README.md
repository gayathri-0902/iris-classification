# Iris Flower Classification

This script utilizes the Iris flower dataset from sklearn to perform classification using Logistic Regression.

## Key Libraries

- pandas
- numpy
- matplotlib.pyplot
- seaborn
- sklearn

## Data

The script utilizes the Iris flower dataset, a benchmark dataset in machine learning for classification. It consists of 150 samples from three Iris species (Setosa, Versicolor, and Virginica) with four features: sepal length, sepal width, petal length, and petal width.

## Script Steps

### Load Libraries and Dataset

- Necessary libraries are imported.
- The Iris flower dataset is loaded from sklearn.

### Data Exploration

- The script explores the data using pandas and seaborn libraries.
- This includes examining data keys, feature names, target labels, and creating a Pandas DataFrame.
- Data distribution for each feature is visualized using histograms.
- Relationships between features and target variables are explored using relational plots.
- Pair plots are generated to visualize the distribution of all features.

### Train-Test Split

- The script splits the data into training and testing sets using train_test_split from scikit-learn.

### Data Preprocessing

- The script prepares the data for modeling by separating features and target variables.

### Baseline Model

- A simple manual model is implemented to predict flower species based on a single feature (petal length).

### Logistic Regression Model

- Logistic regression is employed as the primary classification model.
- The model is trained on the training data.
- Model performance is evaluated using accuracy on both the training and validation sets.

### Cross-Validation

- Cross-validation is performed to assess model generalizability using cross_val_score.

### Analyzing Misclassified Points

- The script identifies and analyzes data points that the model misclassified.
- This helps understand potential areas for improvement.

### Model Tuning

- Grid search is performed to find the optimal hyperparameter (regularization parameter) for the Logistic Regression model.

### Final Model Evaluation

- The final model is trained using the entire training data.
- The model's performance is evaluated on the unseen test set.
- A confusion matrix is generated to visualize the model's performance on the test set.

### Conclusion

- The script concludes by reporting the final test set accuracy achieved by the Logistic Regression model.
