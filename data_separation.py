import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('hand_landmarks_valid.csv')
#print(data)

#splitting into different sets for training, testing and validating
#weighting of each is train approx 70%, test 20% and val 10% of data set
train_val, test = train_test_split(data, test_size=0.2, random_state=42)
train, val = train_test_split(train_val, test_size=0.125, random_state=42)

print(len(train))
print(len(test))
print(len(val))

#setting training label and data
X_train = train.drop(['gesture'],axis=1).to_numpy()
y_train = train['gesture'].to_numpy()

#setting testing label and data
X_test = test.drop(['gesture'],axis=1).to_numpy()
y_test = test['gesture'].to_numpy()

#setting validating label and data
X_val = val.drop(['gesture'],axis=1).to_numpy()
y_val = val['gesture'].to_numpy()
