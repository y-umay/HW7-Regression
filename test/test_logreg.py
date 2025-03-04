"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

'''
To run python -m pytest -v test/* 
'''

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
    model = LogisticRegressor(num_feats=2)
    
    # Simple test case: Predicting for X = [[0, 0]]
    X_test = np.array([[0, 0, 1]])  # Add bias term (1)
    model.W = np.array([0.5, -0.5, 0.0])  # Example weights

    y_pred = model.make_prediction(X_test)
    
    # Manually compute sigmoid(0) = 1 / (1 + e^0) = 0.5
    expected = 1 / (1 + np.exp(0))
    
    assert np.isclose(y_pred, expected, atol=1e-5), f"Expected {expected}, got {y_pred}"


def test_loss_function():
    model = LogisticRegressor(num_feats=1)
    
    # Test inputs: 2 samples with true labels and predictions
    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])  # Predictions close to true labels
    
    loss = model.loss_function(y_true, y_pred)
    
    # Compute expected loss manually
    expected_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    assert np.isclose(loss, expected_loss, atol=1e-5), f"Expected {expected_loss}, got {loss}"


def test_gradient():
    model = LogisticRegressor(num_feats=2)

    # Fake data: X has 2 features (+1 for bias), and 3 samples
    X_test = np.array([
        [1, 2, 1],  # Bias term included
        [3, 4, 1],
        [5, 6, 1]
    ])
    
    # True labels
    y_true = np.array([1, 0, 1])

    # Fake weights for testing
    model.W = np.array([0.2, -0.3, 0.1])

    # Compute gradient
    grad = model.calculate_gradient(y_true, X_test)
    
    # Manually compute predicted values
    y_pred = model.make_prediction(X_test)
    
    # Compute expected gradient: (1/N) * X.T (y_pred - y_true)
    expected_grad = np.dot(X_test.T, (y_pred - y_true)) / len(y_true)

    assert np.allclose(grad, expected_grad, atol=1e-5), f"Expected {expected_grad}, got {grad}"


def test_training():
    model = LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=5, batch_size=1)

    # Small dataset
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])

    # Add bias column
    # X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

    # Store initial weights
    initial_weights = model.W.copy()
    
    # Train model
    model.train_model(X_train, y_train, X_train, y_train)  # Using same data for validation

    # Assert that weights have changed
    assert not np.allclose(initial_weights, model.W), "Weights did not update during training"
