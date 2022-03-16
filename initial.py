import pandas as pd
import numpy as np


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
print(train.info())
#print(test.info())



# main preprocessing procedure:
# 1. drop/replace/impute null values
# 2. feature engineering
# 3. drop any useless variables/columns like passengerId
# 4. categorical transformations (like OneHotEncoding)
# 5. Normalize/Standardize numerical data

# might need to follow some of these steps on the testing set as well

train['HomePlanet'].fillna(train['HomePlanet'].mode(), inplace=True)
train['CryoSleep'].fillna(train['CryoSleep'].mode(), inplace=True)
train['Cabin'].fillna(train['Cabin'].mode(), inplace=True)
train['Destination'].fillna(train['Destination'].mode(), inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['VIP'].fillna(train['VIP'].mode(), inplace=True)
train['RoomService'].fillna(train['RoomService'].mean(), inplace=True)
train['FoodCourt'].fillna(train['FoodCourt'].mean(), inplace=True)
train['ShoppingMall'].fillna(train['ShoppingMall'].mean(), inplace=True)
train['Spa'].fillna(train['Spa'].mean(), inplace=True)
train['VRDeck'].fillna(train['VRDeck'].mean(), inplace=True)


train['cabin_prefix'] = train.Cabin.apply(lambda x: 'na' if pd.isna(x) else x.split('/')[0])
train['cabin_suffix'] = train.Cabin.apply(lambda x: 'na' if pd.isna(x) else x.split('/')[2])
train['passengerId_suffix'] = train.PassengerId.apply(lambda x: 'na' if pd.isna(x) else x.split('_')[1])




drops = ['PassengerId', 'Cabin', 'Name']
train = train.drop(columns=drops)


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# gonna use the dummies method anyway

encoder = OneHotEncoder()
train = encoder.fit_transform(train)


# Normalizing/Standardizing the dataset
from sklearn.preprocessing import StandardScaler, Normalizer
