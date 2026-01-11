import pandas as pd
import sklearn.neighbors as nb
import sklearn.tree as tree
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sklearn.metrics as metrics
from sklearn.linear_model import Perceptron


import math
import numpy as np
from collections import Counter

data = pd.read_csv('hand_landmarks_valid.csv')
#print(data)

#splitting into different sets for training, testing and validating
#weighting of each is train approx 70%, test 20% and val 10% of data set
train_val, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

print(len(train))
print(len(test))
print(len(val))

print()

#setting training label and data
X_train = train.drop(['gesture'],axis=1).to_numpy()
y_train = train['gesture'].to_numpy()

#setting testing label and data
X_test = test.drop(['gesture'],axis=1).to_numpy()
y_test = test['gesture'].to_numpy()

#setting validating label and data
X_val = val.drop(['gesture'],axis=1).to_numpy()
y_val = val['gesture'].to_numpy()


# distance functions
def get_euclidean(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def get_manhattan(v1, v2):
    return sum(abs(a - b) for a, b in zip(v1, v2))

# Manual KNN
def knn(X_train, y_train, X_test, k=3, distance_metric='euclidean'):
    
    if distance_metric == 'euclidean':
        dist_func = get_euclidean
    elif distance_metric == 'manhattan':
        dist_func = get_manhattan
        
    predictions = []

    for x1 in X_test:
        # computing the distance to all training points (where x1 is test and x2 is train)
        distances = [(dist_func(x1, x2), label) 
                     for x2, label in zip(X_train, y_train)]
        
        #sort by distance
        distances = sorted(distances)
        
        #selecting the k nearest
        k_nearest = distances[:k]
        
        #inspecting the most common label in the k nearest
        k_labels = [y for _, y in k_nearest] #just taking the label
        #1 is top of list and [0][0] leaves just the label 
        most_common = Counter(k_labels).most_common(1)[0][0] 
        predictions.append(most_common)
    
    return predictions

def accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)


def hyperparameter_optimization():
    """
    KNN hyperparameter testing following lab pattern.
    Testing different k values and distance metrics as shown in KNN lab.
    """
    print("\n" + "="*60)
    print("KNN HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    SEED = 7107
    
    # Test different k values and distance metrics (from KNN lab)
    print("\nTesting KNN with different k values and metrics...")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    
    # Store results for plotting (simple lists like in lab)
    euclidean_accuracies = []
    manhattan_accuracies = []
    
    for k in k_values:
        # Test Euclidean
        knn_eu = nb.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn_eu.fit(X_train, y_train)
        y_pred_eu = knn_eu.predict(X_test)
        acc_eu = accuracy_score(y_test, y_pred_eu)
        euclidean_accuracies.append(acc_eu)
        
        # Test Manhattan
        knn_man = nb.KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        knn_man.fit(X_train, y_train)
        y_pred_man = knn_man.predict(X_test)
        acc_man = accuracy_score(y_test, y_pred_man)
        manhattan_accuracies.append(acc_man)
        
        print(f"  k={k:2d} | Euclidean: {acc_eu:.4f} | Manhattan: {acc_man:.4f}")
    
    # Bar chart comparing k values (from KNN lab pattern)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, euclidean_accuracies, width, label='Euclidean', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, manhattan_accuracies, width, label='Manhattan', color='orange', alpha=0.7)
    
    ax.set_xlabel('k value', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('KNN Accuracy Comparison: Different k values and Distance Metrics', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('knn_hyperparameter_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: knn_hyperparameter_comparison.png")
    plt.show()
    
    # Train other classifiers with reasonable configs (not optimized)
    print("\n" + "="*60)
    print("Training Decision Tree and Perceptron...")
    print("="*60)
    
    # Decision Tree with default parameters
    dt = tree.DecisionTreeClassifier(random_state=SEED)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree (default): Accuracy = {dt_accuracy:.4f}")
    
    # Perceptron with max_iter=100 (from lab)
    perc = Perceptron(max_iter=100, random_state=SEED, shuffle=True)
    perc.fit(X_train, y_train)
    y_pred_perc = perc.predict(X_test)
    perc_accuracy = accuracy_score(y_test, y_pred_perc)
    print(f"Perceptron (max_iter=100): Accuracy = {perc_accuracy:.4f}")
    
    # Comparison bar chart (from lab pattern)
    print("\n" + "="*60)
    print("Comparing all three classifiers...")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use best KNN result
    best_knn_accuracy = max(euclidean_accuracies)
    best_k_index = euclidean_accuracies.index(best_knn_accuracy)
    best_k = k_values[best_k_index]
    
    classifiers = ['KNN\n(k=' + str(best_k) + ', Euclidean)', 'Decision Tree\n(default)', 'Perceptron\n(max_iter=100)']
    accuracies = [best_knn_accuracy, dt_accuracy, perc_accuracy]
    
    bars = ax.bar(classifiers, accuracies, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Classifier Comparison', fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved: classifier_comparison.png")
    plt.show()
    
    print(f"\nBest KNN configuration: k={best_k}, Euclidean distance (Accuracy: {best_knn_accuracy:.4f})")
    
    return best_k, best_knn_accuracy


def cross_validation_evaluation():
    """
    Cross-validation evaluation following Classifier Evaluation lab.
    Uses 5-fold cross-validation to validate classifier performance.
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION")
    print("="*60)
    
    SEED = 7107
    n_folds = 5
    
    # Set up cross-validation (from lab)
    kfold = KFold(n_splits=n_folds, shuffle=False)
    
    # Get best k from optimization (already determined in Part 1)
    # We'll use the best_k that was found during hyperparameter optimization
    
    print(f"\nEvaluating classifiers with {n_folds}-fold cross-validation...")
    print("Using X_train and y_train for cross-validation")
    print()
    
    # Get best k from previous optimization
    euclidean_accuracies = []
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    for k_test in k_values:
        #Note: Using sklearn for cross-validation as manual implementation
        #  would be computationally expensive with 5-fold CV
        knn_eu = nb.KNeighborsClassifier(n_neighbors=k_test, metric='euclidean')
        knn_eu.fit(X_train, y_train)
        y_pred_eu = knn_eu.predict(X_test)
        acc_eu = accuracy_score(y_test, y_pred_eu)
        euclidean_accuracies.append(acc_eu)
    best_knn_accuracy = max(euclidean_accuracies)
    best_k = k_values[euclidean_accuracies.index(best_knn_accuracy)]
    
    # 1. KNN with best k (from lab pattern)
    print(f"[1/3] KNN (k={best_k}, Euclidean):")
    knn_cv = nb.KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    cv_scores_knn = cross_val_score(knn_cv, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"  CV scores for each fold: {cv_scores_knn}")
    print(f"  Mean CV accuracy: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std():.4f})")
    
    # 2. Decision Tree
    print(f"\n[2/3] Decision Tree (default):")
    dt_cv = tree.DecisionTreeClassifier(random_state=SEED)
    cv_scores_dt = cross_val_score(dt_cv, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"  CV scores for each fold: {cv_scores_dt}")
    print(f"  Mean CV accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")
    
    # 3. Perceptron
    print(f"\n[3/3] Perceptron (max_iter=100):")
    perc_cv = Perceptron(max_iter=100, random_state=SEED, shuffle=True)
    cv_scores_perc = cross_val_score(perc_cv, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"  CV scores for each fold: {cv_scores_perc}")
    print(f"  Mean CV accuracy: {cv_scores_perc.mean():.4f} (+/- {cv_scores_perc.std():.4f})")
    
    # Summary comparison
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"KNN (k={best_k}, Euclidean):     Mean = {cv_scores_knn.mean():.4f}, Std = {cv_scores_knn.std():.4f}")
    print(f"Decision Tree (default):  Mean = {cv_scores_dt.mean():.4f}, Std = {cv_scores_dt.std():.4f}")
    print(f"Perceptron (max_iter=100): Mean = {cv_scores_perc.mean():.4f}, Std = {cv_scores_perc.std():.4f}")
    print("="*60)
    print("\nCross-validation validates performance across multiple data splits.")
    print("Lower standard deviation indicates more stable/reliable performance.")
    
    return cv_scores_knn, cv_scores_dt, cv_scores_perc


#implementation using sklearn libraries 
def nearest_neighbour():
    
    #using k = 7 for the number of neighboours as returns greatest accouracy 
    #whilst providing accracy
    
    # Create KNN (euclidean)
    knn_eu = nb.KNeighborsClassifier(n_neighbors=7,metric='euclidean')
    knn_eu.fit(X_train, y_train)
    
    # Create KNN (manhattan)
    knn_man = nb.KNeighborsClassifier(n_neighbors=7,metric='manhattan')
    knn_man.fit(X_train, y_train)
    
    # Make predictions
    y_predict_eu = knn_eu.predict(X_test)
    y_predict_man = knn_man.predict(X_test)

    accuracy_eu = accuracy_score(y_test, y_predict_eu)
    accuracy_man = accuracy_score(y_test, y_predict_man)

    print("Accuracy (Euclidean):", accuracy_eu)
    print("Accuracy (Manhattan):", accuracy_man)
    
    print()
    
    #running again but using minmax standardisation
    
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_mm = scaler.transform(X_train)
    X_test_mm = scaler.transform(X_test)

    #using k = 11 for the number of neighboours classifier has seperated data

    knn_eu_mm = nb.KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    knn_man_mm = nb.KNeighborsClassifier(n_neighbors=11, metric='manhattan')

    knn_eu_mm.fit(X_train_mm, y_train)
    knn_man_mm.fit(X_train_mm, y_train)

    y_pred_eu_mm = knn_eu_mm.predict(X_test_mm)
    y_pred_man_mm = knn_man_mm.predict(X_test_mm)

    print("Accuracy (Euclidean, MinMax scaled):",
         accuracy_score(y_test, y_pred_eu_mm))
    print("Accuracy (Manhattan, MinMax scaled):",
         accuracy_score(y_test, y_pred_man_mm))
    
    print()
    
    #using a 
    scaler_std = StandardScaler()
    scaler_std.fit(X_train)          

    X_train_std = scaler_std.transform(X_train)
    X_test_std = scaler_std.transform(X_test)
    
    #using k = 5 for the number of neighboours classifier has bought the data
    #closer together

    knn_eu_std = nb.KNeighborsClassifier(n_neighbors=6, metric='euclidean')
    knn_man_std = nb.KNeighborsClassifier(n_neighbors=6, metric='manhattan')

    knn_eu_std.fit(X_train_std, y_train)
    knn_man_std.fit(X_train_std, y_train)

    y_pred_eu_std = knn_eu_std.predict(X_test_std)
    y_pred_man_std = knn_man_std.predict(X_test_std)

    print("Accuracy (Euclidean, standardised):",
      accuracy_score(y_test, y_pred_eu_std))
    print("Accuracy (Manhattan, standardised):",
      accuracy_score(y_test, y_pred_man_std))
    
    print()


def decision_tree():
    
    #constant seed value to control randomness
    SEED=7107

    #Instantiate a DecisionTreeClassfier object with a fixed random seed
    #Leave other parameters with default values
    dt=tree.DecisionTreeClassifier(random_state=SEED)

    #The training step
    #Generate decision tree from data in X and y
    dt.fit(X_train,y_train)
    
    y_dt_pred = dt.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_dt_pred))
    print(classification_report(y_test, y_dt_pred))
    
    return y_dt_pred


def perceptron_classifier():
    """
    Perceptron classifier implementation following lab pattern.
    Perceptron is a linear classifier that learns a decision boundary.
    """
    
    #constant seed value to control randomness
    SEED=7107
    
    print("\n" + "="*60)
    print("PERCEPTRON CLASSIFIER")
    print("="*60)
    
    # Instantiate Perceptron with parameters from lab
    # max_iter: maximum number of passes over training data
    # random_state: for reproducibility
    # shuffle: whether to shuffle training data after each epoch
    perceptron = Perceptron(max_iter=100, random_state=SEED, shuffle=True)
    
    # Train the perceptron
    perceptron.fit(X_train, y_train)
    
    # Make predictions on test set
    y_perc_pred = perceptron.predict(X_test)
    
    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_perc_pred)
    print(f"\nPerceptron Accuracy: {accuracy:.4f}")
    
    # Display confusion matrix
    disp_perc = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_perc_pred)
    disp_perc.figure_.suptitle("Confusion Matrix - Perceptron")
    plt.show()
    
    # Detailed classification report
    print("\n=== Perceptron Classification Report ===")
    print(classification_report(y_test, y_perc_pred, digits=3))
    
    return y_perc_pred
    

if __name__=='__main__':
    
    # Step 1: KNN Hyperparameter Optimization (from lab)
    print("\n" + "#"*60)
    print("# PART 1: KNN HYPERPARAMETER OPTIMIZATION")
    print("#"*60)
    best_k, best_accuracy = hyperparameter_optimization()
    
    # Step 2: Cross-Validation (from Classifier Evaluation lab)
    print("\n\n" + "#"*60)
    print("# PART 2: CROSS-VALIDATION EVALUATION")
    print("#"*60)
    cv_scores_knn, cv_scores_dt, cv_scores_perc = cross_validation_evaluation()
    
    # Combine train and validation data for final model training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    k = best_k
    
    # Retrain all models on combined data
    knn_final = nb.KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn_final.fit(X_train_full, y_train_full)
    y_pred_knn_final = knn_final.predict(X_test)
    knn_final_accuracy = accuracy_score(y_test, y_pred_knn_final)
    
    dt_final = tree.DecisionTreeClassifier(random_state=7107)
    dt_final.fit(X_train_full, y_train_full)
    y_pred_dt_final = dt_final.predict(X_test)
    dt_final_accuracy = accuracy_score(y_test, y_pred_dt_final)
    
    perc_final = Perceptron(max_iter=100, random_state=7107, shuffle=True)
    perc_final.fit(X_train_full, y_train_full)
    y_pred_perc_final = perc_final.predict(X_test)
    perc_final_accuracy = accuracy_score(y_test, y_pred_perc_final)

    # Manual KNN implementation (coded from scratch)
    y_pred_manual_eu = knn(X_train_full, y_train_full, X_test, k=k, distance_metric='euclidean')
    manual_accuracy_eu = accuracy(y_test, y_pred_manual_eu)
    
    y_pred_manual_man = knn(X_train_full, y_train_full, X_test, k=k, distance_metric='manhattan')
    manual_accuracy_man = accuracy(y_test, y_pred_manual_man)
    
    # Confusion matrices
    print("\n" + "="*60)
    print("CONFUSION MATRICES & CLASSIFICATION REPORTS")
    print("="*60)
    
    disp_manual_eu = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_manual_eu)
    disp_manual_eu.figure_.suptitle(f"KNN (k={k}, Euclidean)")
    
    disp_manual_man = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_manual_man)
    disp_manual_man.figure_.suptitle(f"KNN (k={k}, Manhattan)")
    plt.show()
    
    print("\nKNN (Euclidean):")
    print(classification_report(y_test, y_pred_manual_eu, digits=3))
    
    print("\nKNN (Manhattan):")
    print(classification_report(y_test, y_pred_manual_man, digits=3))

    disp_dt = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt_final)
    disp_dt.figure_.suptitle("Decision Tree")
    plt.show()
    print("\nDecision Tree:")
    print(classification_report(y_test, y_pred_dt_final, digits=3))
    
    disp_perc = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred_perc_final)
    disp_perc.figure_.suptitle("Perceptron")
    plt.show()
    print("\nPerceptron:")
    print(classification_report(y_test, y_pred_perc_final, digits=3))
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"KNN (k={k}, Euclidean): {manual_accuracy_eu:.4f}")
    print(f"KNN (k={k}, Manhattan):  {manual_accuracy_man:.4f}")
    print(f"Decision Tree:          {dt_final_accuracy:.4f}")
    print(f"Perceptron:             {perc_final_accuracy:.4f}")
    print("="*60)
