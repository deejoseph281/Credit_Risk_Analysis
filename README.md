# Credit_Risk_Analysis

## Overview of Project

Fast Lending wants to use machine learning to predict credit risk. Our project is to utilize different techniques to train and evaluate models and unbalanced classes. We will utilize imbalanced-learn and scikit-lean libaries to build and evaluate models using resampling. 

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, we’ll use a combinatorial approach of over-and undersampling using the SMOTEENN algorithm. Next, we’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once we’re done, we’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

## Deliverable 1: Use Resampling Models to Predict Credit Risk

  * Create the training variables by converting the string values into numerical ones using the get_dummies() method.
  * Create the target variables.
  * Check the balance of the target variables.

Next, begin resampling the training data. First, use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling ClusterCentroids algorithm to resample the data. For each resampling algorithm, do the following:

  * Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
  * Calculate the accuracy score of the model.
  * Generate a confusion matrix.
  * Print out the imbalanced classification report.
  
## Objective 2: Use the SMOTEENN Algorithm to Predict Credit Risk

  * Using the information we have provided in the starter code, resample the training data using the SMOTEENN algorithm.
  * After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
  * Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

## Objective 3: Use Ensemble Classifiers to Predict Credit Risk

  * Create the training variables by converting the string values into numerical ones using the get_dummies() method.
  * Create the target variables.
  * Check the balance of the target variables.

  * Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
  * After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
  * Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.

  * Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
  * After the data is resampled, calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.

## Written Report on the Credit Risk Analysis

There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

### Random Over Sampler (Naïve Radom Oversampling

 * Balance Accuracy Score: 0.6835
 * Model: Logistic Regression

![image](https://user-images.githubusercontent.com/115019829/219517854-d5cf6c22-d45a-437f-a55e-04821ecfc4d6.png)


### SMOTE Oversampling

 * Balance Accuracy Score: 0.6277
 * Model: Logistic Regression
 
![image](https://user-images.githubusercontent.com/115019829/219517887-a5b41e82-3216-47a2-853f-b26515f0a5db.png)


### Cluster Centroids

 * Balance Accuracy Score: 0.5297
 * Model: Logistic Regression
 
![image](https://user-images.githubusercontent.com/115019829/219517942-e1588667-ae2b-44c5-aa77-66f4fc2eea1d.png)


### SMOTEENN

 * Balance Accuracy Score: 0.6548
 * Model: Logistic Regression

![image](https://user-images.githubusercontent.com/115019829/219518047-ae29947c-a5ad-40ed-b5f4-607bc4086d87.png)


### Balanced Random Forest Classifier

 * Balance Accuracy Score: 0.8731
 * Model: Accuracy Score

![image](https://user-images.githubusercontent.com/115019829/219518132-ed6342a9-0788-4ad5-bf85-1f0130576c90.png)


### Easy Ensemlbe AdaBoost Classifier

 * Balance Accuracy Score: 0.9424
 * Model: Accuracy Score

![image](https://user-images.githubusercontent.com/115019829/219518158-aa5850c8-bff6-4817-a33b-2267ea5f2c03.png)


## SUMMARY

 * The summary of the results:
 
 ![image](https://user-images.githubusercontent.com/115019829/219518224-e31d5ce3-28ea-4725-b51a-20ae12b1e27e.png)

## RECOMMENDATIONS

The Random Over Sampler (Naive Radom Oversampling), SMOTE Oversampling, Cluster Centroids, and Cluster Centroids resulted in a low F1 score, all below 0.02. We can conclude that considering a helpful method for pondering the F1 score, a pronounced imbalance between sensitivity and accuracy will yield a low F1 score.

Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier has high results of precision (pre), recall (rec), specificity (spe), F1-score (f1), geo (Geometric Mean), Index Balanced Accuracy (iba) and support (sup) when we compared with others models.

The Easy Ensemble AdaBoost Classifier has high results regarding the metrics for measuring the performance of imbalanced classes. Also, this model has the highest balance accuracy score with 0.9424. It means that it has the highest exactness of data analysis or includes the correct forecast in Python Scikit learn, so we recommend this model.

