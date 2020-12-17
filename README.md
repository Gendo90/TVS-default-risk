# TVS Default Risk Predictor
## Introduction

This project was developed to gain experience with machine learning using real-world data, and serves as a feasibility analysis of using ML with vehicle loan and customer information from TVS - a large Indian motorcycle company. The raw data is available [here](https://www.kaggle.com/sjleshrac/tvs-loan-default) and is sourced from [Kaggle.com](www.kaggle.com). Our project addresses the concern of cross-selling personal loans to existing auto loan customers who are possible default risks, because personal loans are unsecured and therefore carry a relatively high cost for the provider if a default occurs. Using ML, a subset of the current vehicle loan customer base could be selected that only includes customers unlikely to default on a personal loan, thus reducing the risk to the personal loan provider (in this particular case, TVS Credit).

## ML Models Used

To start identifying high-impact variables on default risk, a random forest was constructed using the [sklearn](https://scikit-learn.org/stable/) [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest%20classifier#sklearn.ensemble.RandomForestClassifier) in order to determine the [important features](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) in the dataset from the feature_importances_ attribute. The features that had relatively high impact on the "default" column result were then used for all the ML models developed from this dataset.

### Logistic Classification

### Decision Tree

### Random Forest

### Gradient-Boosted Trees (GB)

#### sklearn

#### XGBoost

## Findings

## Planned Features