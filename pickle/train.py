# import all the libraries

import pandas as pd  # pip install pandas
from sklearn import model_selection # pip install scikit-learn
from sklearn.linear_model import LogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv(url , names = names)
print(df)

X = df.iloc[: , :8]
y = df.iloc[:, 8]

X_train , X_test, y_train, y_test = model_selection.train_test_split(X , y , test_size = 0.2 , random_state = 101)

# train the model
model = LogisticRegression()
model.fit(X_train , y_train,)

# accuracy
result = model.score(X_test , y_test)

print(result)


# save the model (.sav  / .pkl)
pickle.dump(model , open('diabities.pkl' , 'wb'))