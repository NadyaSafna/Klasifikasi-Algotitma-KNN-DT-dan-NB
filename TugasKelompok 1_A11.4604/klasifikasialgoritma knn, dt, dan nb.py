#impor library yang digunakan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#impor dataset irir yang tersedia di library
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

#membagi dataset menjadi data latih dan data uji dengan rasio 70:30.
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#menerapkan algoritma KNN, Decision Tree, dan Naive Bayes pada data latih.
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)

nb = GaussianNB()
nb.fit(X_train, y_train)

#melakukan prediksi pada data uji menggunakan masing-masing algoritma.
knn_y_pred = knn.predict(X_test)

dt_y_pred = dt.predict(X_test)

nb_y_pred = nb.predict(X_test)


#menghitung akurasi prediksi dari ketiga algoritma.
print("Akurasi KNN:", accuracy_score(y_test, knn_y_pred))

print("Akurasi Decision Tree:", accuracy_score(y_test, dt_y_pred))

print("Akurasi Naive Bayes:", accuracy_score(y_test, nb_y_pred))
