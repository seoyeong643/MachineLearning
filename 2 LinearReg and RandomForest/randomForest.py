# Load libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Import Random Forest Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for calculating the various required metrics calculation
import matplotlib.pyplot as plt

# Read the dataset and look at the head to gauge the column names and whether column names actually exists

df_rf = pd.read_csv('diabetes.csv') # YOUR CODE HERE

df_rf.head()

# Perform the basic Null Check to decide whether imputation or drop is required

df_rf.isnull().sum()

# Split the data into features --> X and target values --> y
# Use Outcome as the target variable (i.e. y variable) and the rest as features (i.e. X variable)

X = df_rf.drop(columns='Outcome') # YOUR CODE HERE
y = df_rf['Outcome'] # YOUR CODE HERE
print(X)

# Perform a train test split with 70 - 30 train - test split
# Use random state as 0 but you are encouraged to play around with the values of random state as well

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0) # YOUR CODE HERE

# Create Random Forest classifier object - change hyper params and get different results maybe answer with best values

# Play around with the values of the RandomForestClassifier - change criterion and max_depth to find the best f1 score
# Prescribed values - criterion = "gini", min_samples_leaf = 1, min_samples_split = 10, max_features='auto', random_state=1

rf_clf = RandomForestClassifier(criterion="gini", min_samples_leaf=1, min_samples_split=10, max_features='auto', random_state=1) # YOUR CODE HERE

# Train and fit Random Forest Classifer
rf_clf.fit(X_train, y_train) # YOUR CODE HERE

# Predict the response for test dataset
y_pred = rf_clf.predict(X_test) # YOUR CODE HERE

# Report the accuracy, precision, recall and f1_score into the variables labelled thus - you may try out various combinations referring here: https://scikit-learn.org/stable/modules/classes.html#classification-metrics

rf_y_true = y_test

accuracy = metrics.accuracy_score(rf_y_true, y_pred)    # Put the value of accuracy here -- 1.1 of Gradescope tests
precision = metrics.precision_score(rf_y_true, y_pred, average='weighted')  # Put the value of precision here -- 1.2 of Gradescope tests
recall = metrics.recall_score(rf_y_true, y_pred, average='weighted')        # Put the value of recall here -- 1.3 of Gradescope tests
f1_score = metrics.f1_score(rf_y_true, y_pred, average='weighted')          # Put the value of F1 score here -- 1.4 of Gradescope tests

print("Accuracy:", accuracy) # -- 1.1
print("Precision Score:", precision) # -- 1.2
print("Recall Score: ", recall) # -- 1.3
print("F1 Score: ", f1_score) # -- 1.4

# Here a template for writing the function for getting the tpr, fpr and threshold as well as AUC values is given as well as a simple way to plot the ROC curve
# you need to find out the predicted probabilities for the classifier you have written - using: redict_proba() function
# Pass these probabilities along with the y_true values into this function as specified and find the values mentioned above
# and report the AUC value and plot the ROC curve

from sklearn.metrics import roc_curve, auc

rf_auc = 0

def plot_roc(rf_y_true, rf_probs):
    
    # Use sklearn.metrics.roc_curve() to get the values based on what the funciton returns - Read documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    rf_fpr, rf_tpr, rf_threshold = metrics.roc_curve(rf_y_true, rf_probs) # YOUR CODE HERE

    # Use sklearn.metrics.auc() to get the AUC score of your model - Read documentation here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
    rf_auc_val = metrics.auc(rf_fpr, rf_tpr) # Find out and put the AUC score here -- 1.5 of Gradescope tests
    
    # Report the AUC score for the model you have created
    print('AUC=%0.2f'%rf_auc_val) # -- 1.5
    
    # Plot the ROC curve using the probabilities and the true y values as passed into the fuction we have defined here

    plt.plot(rf_fpr, rf_tpr, label = 'AUC=%0.2f'%rf_auc_val, color = 'darkorange')
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return rf_auc_val

# We pass in the following parameters to the custom function we have written. For details as to why we do this - Read here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

rf_probs = rf_clf.predict_proba(X_test) [:,1] # This gives the classifiers' prediction probabilities for each class for the predicted samples
rf_auc = plot_roc(rf_y_true, rf_probs) # This is where 1.5 is graded in Gradescope

# Report which cross validation value of k gave the best result as well as the accuracy score - use StrafiedKFold Cross Validation method - Read about it here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):

    skfold = StratifiedKFold(n_splits=k, random_state=100, shuffle=True) # YOUR CODE HERE FOR: StratifiedKFold() - use random_state = 100, shuffle = True
    results_skfold_acc = (cross_val_score(rf_clf, X, y, cv = skfold)).mean() * 100.0
    
    if results_skfold_acc > max_acc:# conditional check for getting max value and corresponding k value
        max_acc = results_skfold_acc
        max_k = k
    	# YOUR CODE HERE

    print("Accuracy: %.2f%%" % (results_skfold_acc))

best_accuracy = max_acc # Put the accuracy score here from the values that you got -- 1.6 of Gradescope tests
best_k_fold = max_k     # Put the value of k that gives the best accuracy here # -- 1.7 of Gradescope tests

print(best_accuracy, best_k_fold)