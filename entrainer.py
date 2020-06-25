import numpy as np
import sys
import load_datasets
import knn

train_ratio=80

model_knn=classifieur.Knn()

train_iris, train_labels_iris, test_iris, test_labels_iris = load_datasets.load_iris_dataset(train_ratio)

print("-------Apprentissage avec KNN---------\n\n")

# Entrainez votre classifieur

K_iris=model_knn.best_k(train_iris,train_labels_iris, model_knn.euclidienne)
print("choix de k_iris:\n",K_iris)

print("TRAINING IRIS KNN \n\n")
training_iris_knn=model_knn.train(train_iris, train_labels_iris, model_knn.euclidienne, K_iris)

# Tester votre classifieur

print("TEST IRIS KNN \n\n")
tester_iris_knn=model_knn.test(train_iris, train_labels_iris, test_iris, test_labels_iris, model_knn.euclidienne, K_iris)

