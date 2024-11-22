from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import KFold
import numpy as np


class Model:
  def __init__(self, data):
    self.data = data 
    
  def train_SVC(self):
    svc = SVC()
    # self.__cross_validate(svc)
    self.model = svc.fit(self.data.lda_3, self.data.encoded_moves)
    return svc
    
  def train_Decision_Tree(self):
    dtc = DecisionTreeClassifier()
    self.__cross_validate(dtc)
    self.model = dtc.fit(self.data.lda_3, self.data.encoded_moves)
    return dtc
  
  def train_neural_network(self):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True)
    
    inputs = self.data.lda_3
    outputs = self.data.encoded_moves

    # K-fold Cross Validation model evaluation
    fold_no = 1
    scores = []
    for train, test in kfold.split(inputs, outputs):
      inputs_train = inputs[train].reshape(-1, 3)  # Each sample has 3 features
      inputs_test = inputs[test].reshape(-1, 3)  # Each sample has 3 features

      # Define the model architecture
      model = Sequential()
      model.add(Input(shape=(3,)))  # Input shape is (3,) since each sample has 3 features
      model.add(Dense(256, activation='relu'))
      model.add(Dense(128, activation='relu'))
      model.add(Dense(164, activation='relu'))
      model.add(Dense(16, activation='relu'))
      model.add(Dense(4, activation='softmax'))  # Output layer

      # Compile the model
      model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

      # Fit the model
      history = model.fit(inputs_train, outputs[train], epochs=10, batch_size=32)
      # Evaluate the model
      score = model.evaluate(inputs_test, outputs[test], verbose=0)
      scores.append(score[1])
      print(f"Fold {fold_no} - Test loss: {score[0]}, Test accuracy: {score[1]}")
      fold_no += 1
    
    self.model = Sequential()
    model.add(Input(shape=(3,)))  # Input shape is (3,) since each sample has 3 features
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(164, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(inputs_train, outputs[train], epochs=10, batch_size=32)
    print("average accuracy = " + str(sum(scores)/len(scores)))
  
  def __cross_validate(self, model):
    results = cross_validate(model, self.data.lda_3, self.data.encoded_moves, cv=10, scoring='balanced_accuracy')
    print(results)
    print(f'Average accuracy: {results["test_score"].mean()}')
    
