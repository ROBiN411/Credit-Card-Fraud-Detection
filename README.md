# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect fraudulent credit card transactions using the Credit Card Fraud Detection Dataset from Kaggle. The goal is to accurately identify fraudulent transactions to minimize financial losses.

## About the Project

This project is focused on detecting fraudulent credit card transactions using machine learning techniques. The goal is to build a model that can accurately identify fraudulent transactions from a dataset of credit card transactions.

The steps involved in this project include:

Data Loading and Exploration: Loading the credit card transaction dataset and exploring its characteristics, including the distribution of fraudulent vs. non-fraudulent transactions.
Data Analysis and Visualization: Performing data analysis and visualizing key aspects of the data, such as correlations between features.
Feature Importance and Selection: Identifying and selecting the most important features for the model.
Data Splitting: Splitting the data into training and testing sets for model development and evaluation.
Model Selection and Training: Choosing and training appropriate machine learning models, such as Decision Trees and Random Forests.
Hyperparameter Tuning: Optimizing the model's performance through hyperparameter tuning.
Model Evaluation: Evaluating the trained model using metrics like accuracy and confusion matrix.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project requires Python and the following libraries:

* numpy
* pandas
* matplotlib.pyplot
* seaborn
* re
* sklearn.model_selection (for train_test_split, KFold, StratifiedKFold, RandomizedSearchCV)
* sklearn.metrics (for accuracy_score, confusion_matrix)
* sklearn.preprocessing (for StandardScaler)
* sklearn.tree (for DecisionTreeClassifier)
* sklearn.ensemble (for RandomForestClassifier, ExtraTreesClassifier)
* kagglehub
* os

### Installation

1.  Clone the repo
    ```bash
    git clone https://github.com/your_username/your_project_name.git
    ```
2.  Navigate to the project directory
    ```bash
    cd your_project_name
    ```
3.  Install the required libraries
    ```bash
    pip install tensorflow keras numpy pandas matplotlib seaborn
    ```

## Usage

You can run the code directly in a Google Colab or in a Jupyter Notebook. The notebook `Credit-Card-Fraud-Detection.ipynb` contains the complete code and visualizations.

1.  Open the notebook in a Google Colab (like Jupyter Notebook, or JupyterLab).
2.  Run the cells sequentially to load the data, build the model, train it, and visualize the results.

## Model Architecture

Here I'm using two common machine learning models for classification:

* Decision Tree Classifier
  
A Decision Tree Classifier is a non-parametric supervised learning algorithm used for classification. It works by recursively partitioning the data based on the values of the features. The architecture can be visualized as a tree-like structure where:

Nodes: Represent a test on an attribute (feature).
Branches: Represent the outcome of the test.
Leaves: Represent the class label (fraudulent or non-fraudulent).
The tree is built by selecting the best attribute to split the data at each node, typically using criteria like Gini impurity or information gain.

* Random Forest Classifier
  
A Random Forest Classifier is an ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. The architecture involves:

Multiple Decision Trees: The forest consists of a large number of individual decision trees that operate as an ensemble.
Random Subset of Features: Each tree in the random forest is built from a random subset of the features.
Bagging (Bootstrap Aggregating): Each tree is trained on a random subset of the training data, sampled with replacement.
By combining the predictions of multiple trees, the random forest reduces overfitting and improves the model's robustness and accuracy compared to a single decision tree.

## Results

The dataset contains a significant imbalance, with only a small percentage of fraudulent transactions. Feature importance analysis identified key features like V17, V12, and V14.
A Random Forest Classifier was trained and achieved a high accuracy of 0.9994. However, due to the data imbalance, it's important to also consider metrics such as precision, recall, and the confusion matrix for a complete evaluation of the model's ability to detect fraud.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request


Your Name - \[Your Email]
Project Link: \[Your Repository Link]
