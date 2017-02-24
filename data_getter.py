import numpy as np
import random

def get_raw_data():
  X_train_PGH_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/train.bin"
  X_train_Orlando_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/train.bin"
  X_train_NYC_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/train.bin"

  X_val_PGH_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/val.bin"
  X_val_Orlando_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/val.bin"
  X_val_NYC_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/val.bin"

  X_train_PGH = np.load(X_train_PGH_path)
  X_train_Orlando = np.load(X_train_Orlando_path)
  X_train_NYC = np.load(X_train_NYC_path)

  X_val_PGH = np.load(X_val_PGH_path)
  X_val_Orlando = np.load(X_val_Orlando_path)
  X_val_NYC = np.load(X_val_NYC_path)

  return X_train_PGH, X_train_Orlando, X_train_NYC, X_val_PGH, X_val_Orlando, X_val_NYC

def randomize_data(X, y):
  rand_sequence = random.sample(range(0,X.shape[0]),X.shape[0])
  randomized_X = X[rand_sequence,]
  randomized_y = y[rand_sequence,]

  return randomized_X, randomized_y


def get_preapared_data():
  X_train_PGH, X_train_Orlando, X_train_NYC, X_val_PGH, X_val_Orlando, X_val_NYC = get_raw_data()

  y_train_PGH = np.full(X_train_PGH.shape[0], 0, dtype=np.int)
  y_train_Orlando = np.full(X_train_Orlando.shape[0], 1, dtype=np.int)
  y_train_NYC = np.full(X_train_NYC.shape[0], 2, dtype=np.int)

  y_val_PGH = np.full(X_val_PGH.shape[0], 0, dtype=np.int)
  y_val_Orlando = np.full(X_val_Orlando.shape[0], 1, dtype=np.int)
  y_val_NYC = np.full(X_val_NYC.shape[0], 2, dtype=np.int)

  X_train_full = np.vstack([X_train_PGH, X_train_Orlando, X_train_NYC])
  y_train_full = np.hstack([y_train_PGH, y_train_Orlando, y_train_NYC])

  X_val_full = np.vstack([X_val_PGH, X_val_Orlando, X_val_NYC])
  y_val_full = np.hstack([y_val_PGH, y_val_Orlando, y_val_NYC])

  X_train_random, y_train_random = randomize_data(X_train_full, y_train_full)
  X_val_random, y_val_random = randomize_data(X_val_full, y_val_full)

  return {"X_train": X_train_random,
          "y_train": y_train_random,
          "X_val": X_val_random,
          "y_val": y_val_random
          }

def get_test_data():
  X_test_PGH_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/PGH/test.bin"
  X_test_Orlando_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/Orlando/test.bin"
  X_test_NYC_path = "/Users/Ankush/Desktop/DeepLearning/Project/data/Sample2/NYC/test.bin"

  X_test_PGH = np.load(X_test_PGH_path)
  X_test_Orlando = np.load(X_test_Orlando_path)
  X_test_NYC = np.load(X_test_NYC_path)

  y_test_PGH = np.full(X_test_PGH.shape[0], 0, dtype=np.int)
  y_test_Orlando = np.full(X_test_Orlando.shape[0], 1, dtype=np.int)
  y_test_NYC = np.full(X_test_NYC.shape[0], 2, dtype=np.int)

  

  return {"X_test_PGH": X_test_PGH,
          "X_test_Orlando": X_test_Orlando,
          "X_test_NYC": X_test_NYC,
          "y_test_PGH": y_test_PGH,
          "y_test_Orlando": y_test_Orlando,
          "y_test_NYC": y_test_NYC,

          }

def get_full_data():
  prepared_data = get_preapared_data()
  test_data = get_test_data()
  return(dict(prepared_data.items() + test_data.items()))


