from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

print(iris.data.shape)
print(iris.target_names)

iris_df = pd.DataFrame(data = np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
        # np.c_ is the numpy concatenate function which is used to concat iris['data'] and iris['target'] arrays

iris_df.groupby('target').size()

X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential()
model.add(Dense(16, activation='relu', input_shape = (4,)))
model.add(Dropout(0.25))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size = 1, epochs = 20, verbose = 2, validation_data = (X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test, verbose = 42)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# test
print("Predicted iris type:", iris.target_names[np.argmax(model.predict([[6.3, 3.3, 5.2, 2.1]]))])




