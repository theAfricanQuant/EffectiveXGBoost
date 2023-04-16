# Effective XGBoost by Matt Harrison - My Personal Take

This repository contains my personal notes, examples, and code implementations based on the book "Effective XGBoost" by Matt Harrison. The purpose of this repo is to document my journey as I peer thru the mind of a master displaying his craft so i could glean some insights myself into building models that I could use for my projects.

## Table of Contents

2 Datasets 
2.1 Cleanup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
2.2 Cleanup Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

3 Exploratory Data Analysis 
3.1 Correlations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
3.2 Bar Plot . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

4 Tree Creation 23
4.1 The Gini Coefficient . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
4.2 Coefficients in Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
4.3 Another Visualization Tool . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

5 Stumps on Real Data 31
5.1 Scikit-learn stump on real data . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
5.2 Decision Stump with XGBoost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
5.3 Values in the XGBoost Tree . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

6 Model Complexity & Hyperparameters 
6.1 Underfit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
6.2 Growing a Tree . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
6.3 Overfitting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
6.4 Overfitting with Decision Trees . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

7 Tree Hyperparameters 
7.1 Decision Tree Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
7.2 Tracking changes with Validation Curves . . . . . . . . . . . . . . . . . . . . . . 
7.3 Leveraging Yellowbrick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
7.4 Grid Search............................................................................

8 Random Forest 51
8.1 Ensembles with Bagging . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
8.2 Scikit-learn Random Forest . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
8.3 XGBoost Random Forest . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
8.4 Random Forest Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
8.5 Training the Number of Trees in the Forest . . . . . . . . . . . . . . . . . . . . . 




