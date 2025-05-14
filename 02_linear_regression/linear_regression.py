import math

import numpy as np


class LinearRegressionModel:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, Y):
        # b = r * Sy/Sx
        n = len(X)
        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        r_numerator = np.sum((X - mean_x) * (Y - mean_y))
        r_denominator = math.sqrt(np.sum((X - mean_x) ** 2) * np.sum((X - mean_x) ** 2))

        r = r_numerator/r_denominator

        sY = math.sqrt(np.sum(Y - mean_y)) / n - 1 # Standard deviation
        sX = math.sqrt(np.sum(X - mean_x)) / n - 1

        self.slope = r * (sY / sX)
        self.intercept = mean_y - self.slope * mean_x

    def predict(self, X):
        """
        Make predictions using the linear regression model.

        X: numpy array or list of values to make predictions for
        Returns: numpy array of predictions
        """
        Y = self.slope * X + self.intercept
        return Y

    def coefficients(self):
        """
        Return the coefficients of the model.

        Returns: Tuple (slope, intercept)
        """
        return (self.slope, self.intercept)

    def r_squared(self, Y_actual, Y_predicted):
        """
        Calculate the R-squared value for the model.

        Y_actual: numpy array or list of actual values
        Y_predicted: numpy array or list of predicted values
        Returns: float representing the R-squared value
        """
        # (Var(mean) - Var(line)) / Var(mean) == 1 - Var(line) / Var(mean)
        # Var is sum of residuals
        # R is correlation
        # Result means "less variation around the line, that the mean

        var_mean = np.sum((Y_actual - np.mean(Y_actual)) ** 2)
        var_line = np.sum((Y_actual - Y_predicted) ** 2)

        return (var_mean - var_line) / var_mean

# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 4, 5, 4, 5])

    # Create and train the model
    model = LinearRegressionModel()
    model.fit(X, Y)

    # Make predictions
    predictions = model.predict(X)

    # Output the coefficients
    print("Slope:", model.coefficients()[0], "Intercept:", model.coefficients()[1])

    # Calculate and print the R-squared value
    print("R-squared:", model.r_squared(Y, predictions))