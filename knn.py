import pandas as pd
import sklearn.neighbors as nb
import sklearn.tree as tree
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import math
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

    tree.plot_tree(dt,node_ids=True)
    plt.title('Decision Tree Classifier')
    plt.show()
    
    y_dt_pred = dt.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_dt_pred))
    print(classification_report(y_test, y_dt_pred))
    

if __name__=='__main__':
        
    k = 10
    #the larger the k the more accurate however longer processing time
    
    print("K =", k)

    y_pred_eu = knn(X_train, y_train, X_test, k=k, distance_metric='euclidean')
    y_pred_man = knn(X_train, y_train, X_test, k=k, distance_metric='manhattan')

    print("KNN Euclidean Accuracy:", accuracy(y_test, y_pred_eu))
    print("KNN Manhattan Accuracy:", accuracy(y_test, y_pred_man))
