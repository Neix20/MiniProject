from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', return_X_y=True, cache=True) # version 0.20 cannot cache, y is string object