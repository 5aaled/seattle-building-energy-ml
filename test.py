from sklearn.datasets import make_regression
X, y = make_regression(n_samples=30, n_features=1, noise=40, random_state=42)
from sklearn.linear_model import Ridge
model = Ridge(alpha=0)
model.fit(X, y)
y_pred = model.predict(X)