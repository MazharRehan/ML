import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression


def gradient_descent(x, y):
    m_current = b_current = 0
    iterations = 1000000
    if len(x) != len(y):
        raise ValueError("The length of x and y must be the same")
    n = len(x)
    learning_rate = 0.0002
    previous_cost = 0

    for i in range(iterations):
        y_predicted = m_current * x + b_current
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        m_current = m_current - learning_rate * md
        b_current = b_current - learning_rate * bd

        if math.isclose(cost, previous_cost, rel_tol=1e-20):
            break
        previous_cost = cost
        print("m {}, b {}, cost {}, iteration {}".format(m_current, b_current, cost, i))

    return m_current, b_current


def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']], df.cs)
    return r.coef_, r.intercept_


if __name__ == '__main__':
    df = pd.read_csv('test_scores.csv')
    x = np.array(df.math)  # or df['math']
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)

    print("Using gradient descent function: Coefficient {} Intercept {}".format(m, b))
    m_sklearn, b_sklearn = predict_using_sklean()
    print("Using sklearn: Coefficient {} Intercept {}".format(m_sklearn, b_sklearn))

"""
m 1.0177381667793246, b 1.9150826134339467, cost 31.604511334602297, iteration 415532
Using gradient descent function: Coefficient 1.0177381667350405 Intercept 1.9150826165722297
Using sklearn: Coefficient [1.01773624] Intercept 1.9152193111568891
"""
