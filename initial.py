import pandas as pd
import numpy as np


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train["train_test"] = 0
test["train_test"] = 1

#print(train['Transported'].value_counts())
#print(train.info())

all_data = pd.concat([train, test])






# main preprocessing procedure:
# 1. drop/replace/impute null values
# 2. feature engineering
# 3. drop any useless variables/columns like passengerId
# 4. categorical transformations (like OneHotEncoding)
# 5. Normalize/Standardize numerical data

# might need to follow some of these steps on the testing set as well
# WE MAY NEED TO COMBINE TRAIN AND TEST SETS SO THAT WHEN NORMALIZING/IMPUTING
# DATA WE TREAT IT AS THE SAME AS THE TRAIN SET

# 1. drop/replace/impute null values
#all_data['HomePlanet'].fillna(all_data['HomePlanet'].mode(), inplace=True)
#all_data['CryoSleep'].fillna(all_data['CryoSleep'].mode(), inplace=True)
#all_data['Cabin'].fillna(all_data['Cabin'].mode(), inplace=True)
#all_data['Destination'].fillna(all_data['Destination'].mode(), inplace=True)
all_data['Age'].fillna(all_data['Age'].mean(), inplace=True)
#all_data['VIP'].fillna(all_data['VIP'].mode(), inplace=True)
all_data['RoomService'].fillna(all_data['RoomService'].mean(), inplace=True)
all_data['FoodCourt'].fillna(all_data['FoodCourt'].mean(), inplace=True)
all_data['ShoppingMall'].fillna(all_data['ShoppingMall'].mean(), inplace=True)
all_data['Spa'].fillna(all_data['Spa'].mean(), inplace=True)
all_data['VRDeck'].fillna(all_data['VRDeck'].mean(), inplace=True)


#print(all_data['HomePlanet'].value_counts())




# 2. feature engineering
all_data['cabin_prefix'] = all_data.Cabin.apply(lambda x: 'na' if pd.isna(x) else x.split('/')[0])
all_data['cabin_suffix'] = all_data.Cabin.apply(lambda x: 'na' if pd.isna(x) else x.split('/')[2])
all_data['passengerId_suffix'] = all_data.PassengerId.apply(lambda x: 'na' if pd.isna(x) else x.split('_')[1])


# 3. drop any useless variables/columns like passengerId
drops = ['PassengerId', 'Cabin', 'Name']
all_data = all_data.drop(columns=drops)


# 4. categorical transformations (like OneHotEncoding)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
cont = ['Age', 'RoomService', 'FoodCourt','ShoppingMall','Spa','VRDeck']
cat = [x for x in all_data.columns if x not in cont]
cat.remove('Transported')
cat.remove('train_test')
#print(f'cont {cont}')
#print(f'cat {cat}')

# gonna use the dummies method anyway
#encoder = OneHotEncoder()
#train = encoder.fit_transform(train)
# will also probably need to encode the labels

all_data_ohe = pd.get_dummies(all_data, columns=cat)


# 5. Normalize/Standardize numerical data
from sklearn.preprocessing import StandardScaler, Normalizer
scale = StandardScaler()
all_data_ohe_scaled = all_data_ohe.copy()
all_data_ohe_scaled[cont] = scale.fit_transform(all_data_ohe_scaled[cont])


# split data again
X_train = all_data_ohe_scaled[all_data_ohe_scaled.train_test == 0].drop(['train_test', 'Transported'], axis=1)
X_test = all_data_ohe_scaled[all_data_ohe_scaled.train_test == 1].drop(['train_test'], axis=1)
y_train = all_data_ohe_scaled[all_data_ohe_scaled.train_test == 0].Transported


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




# Naive Bayes provides a good baseline for classification tasks
nb = GaussianNB()
#cv = cross_val_score(nb, X_train, y_train, cv=5)
#print(cv)
#print(cv.mean())


lr = LogisticRegression(max_iter = 2000)
cv = cross_val_score(lr, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

