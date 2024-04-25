import numpy as np
import numpy as np

# Generate some synthetic data
# Independent variable
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
# Dependent variable
Y = np.array([2, 4, 6, 8, 10])

# Add a column of ones to X to account for the intercept term
X_b = np.c_[np.ones((5, 1)), X]  # Add intercept column

# Calculate the coefficients using the normal equation
beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

# Print the coefficients
print("Coefficients:", beta)
# The first coefficient is the intercept and the second is the slope

# Make predictions
X_new = np.array([[0], [6]])  # Predict values from 0 to 6
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add intercept term
predictions = X_new_b.dot(beta)
print("Predictions:", predictions)
