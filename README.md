PYTHON CODE
import pandas as pd import numpy as np
import matplotlib.pyplot as plt import seaborn as sns from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestRegressor from sklearn import metrics ts=pd.read_csv('TEST SHEET 2.csv')
ts.head() ts.isnull().sum() ts.shape ts.info print(ts.GRADE.value_counts()) ts.replace({'AGE':{'UNKNOWN':-1}},inplace=True) ts.head() ts.replace({'GRADE':{'M25':25,'M30':30,'M35':35}},inplace=True)
X = ts.drop(['STRENGTH FROM','STRENGTH TO'],axis=1)
Y = ts['STRENGTH FROM']
Z = ts['STRENGTH TO']
print(X) print(Y) print(Z) rfg=RandomForestRegressor()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2) rfg.fit(X_train,Y_train) FDB=rfg.predict(X_test) from sklearn.metrics import r2_score errorscore = r2_score(Y_test, FDB) print("R squared error:", errorscore) plt.scatter(Y_test,FDB) plt.xlabel("actual data") plt.ylabel("predicted data") plt.show()
rfg1=RandomForestRegressor() rfg1.fit(X_train,Y_train) FDB22=rfg.predict(X_test) from sklearn.metrics import r2_score
errorscore = r2_score(Y_test, FDB22) print("R squared error:", errorscore) plt.scatter(Y_test,FDB) plt.xlabel("actual data") plt.ylabel("predicted data") plt.show()
grade = input("Enter Grade (M25, M30, M35): ") upv = float(input("Enter UPV: ")) rebound = float(input("Enter Rebound: ")) age = int(input("Enter Age: ")) input_data = {'GRADE': [grade], 'UPV': [upv], 'REBOUND': [rebound], 'AGE': [age]} input_df = pd.DataFrame(input_data) input_df.replace({'AGE': {'UNKNOWN': -1}}, inplace=True) input_df.replace({'GRADE': {'M25': 25, 'M30': 30, 'M35': 35}}, inplace=True) prediction_Y = rfg.predict(input_df) prediction_Z =
rfg1.predict(input_df) print("Predicted Strength From (Y):", prediction_Y) print("Predicted Strength To (Y):", prediction_Z)
