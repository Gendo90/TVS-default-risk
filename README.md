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

Once the above ML models were created, they needed to be evaluated in a standard way, to get an overall "score" of how good the model was for minimizing default risk while making the same or more money from lending overall. Additionally, the data was imbalanced, with two classes (defaulter or non-defaulter) that have a ratio of approximately 1:50 in the dataset provided, corresponding to ~2% default rate. So using a cost function that prioritizes accuracy - that is, the model correctly identifies as many items as possible - has the unfortunate result of simply classifying **all** the loan customers as non-defaulters because this will result in ~98% accuracy rate of the classification model. 

Instead, a deeper dive into the economic costs of loans defaulting vs. being repaid was required. The results were that each default loan was worth approximately 5 non-default loans, given the total amount lost by a default loan and the total profit on a non-default loan - to see the calculations of the loan values and the ratio, see [here](./cleaned_data/Dataset\ Financial\ Exploration\ Notebook.ipynb). This result guided the creation of a cost function of 
```
cost = d1 - 5*d2
```
where `d1` is the number of defaulters correctly identified by the model and `d2` is the number of customers who were identified as a default risk who did not actually default. This function was later modified to increase the coverage of the defaulters by adding an accuracy multiplier (which weights the function so that given the same overall cost, if more defaulters are correctly identified, then the one with greater coverage is "better" to the model). This can be seen as giving the final cost function:
```
cost = (d1/d_total)*(d1 - 5*d2)
```
where `d_total` is the total number of defaulters in the dataset. This equation was particularly useful in guiding the construction fo the random forests and gradient-boosted trees.

The ML models based on the algorithms [above](##ML\ Models\ Used) were all evaluated against one another, and the results of that comparison showed the gradient-boosted trees maximizing the cost function compared to the other models, and therefore giving the greatest economic gain (profit) to the lender. 

## Planned Features

As discussed in the findings, the gradient-boosted trees give the best results for this particular use-case of ML. However, a "better" model (one identifies more potential defaulters at a lower cost of non-defaulters) may be possible using neural networks. The next steps in developing a personal loan customer screening process involve creating a variety of neural networks using different parameters to determine whether or not they make the process more accurate and/or comprehensive.

Additionally, the app itself could be improved by implementing front-end barriers to submitting the form if the input fields are invalid - by using JavaScript and basic input validation techniques.