import copy, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_cost(X, y, w, b):

    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i,:], w) + b
        cost += (f_wb_i - y[i])**2
    
    cost /= (2*m)
    return cost

def compute_gradient(X, y, w, b):

    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w))+b - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i,j]

        dj_db += err
    dj_dw /= m 
    dj_db /= m
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

    J_history = []
    p_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        J_history.append(cost_function(X, y, w, b))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"w: {w} b: {b}")
        p_history.append([w,b])

    return w, b, J_history, p_history

def feature_scaling(X):

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    
    X_scaled = (X - mean) / std_dev
    
    return X_scaled


if __name__ == "__main__":

    data_file_path = Path(__file__).parent / 'data.csv'
    df = pd.read_csv(data_file_path)

    x_train = df['X'].values
    y_train = df['Y'].values

    standard_deviation = np.std(x_train)

    X = np.c_[x_train, x_train**2]
    
    X_scaled = feature_scaling(X)    

    w_in = np.zeros((X.shape[1],))
    b_in = 0
    iterations = 10000
    tmp_alpha = 1e-3

    j_hist = []
    p_hist = []

    model_w, model_b, j_hist, p_hist = gradient_descent(X_scaled, y_train, w_in, b_in, compute_cost, compute_gradient, tmp_alpha, iterations)

    print("Model w: ", model_w)
    print("Model b: ", model_b)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(12,5))

    ax2.scatter(x_train, y_train, marker='x', c='r', label="Actual Value")
    ax1.set_title("Cost vs. iteration"); ax2.set_title("Polynomial Model")
    ax1.set_xlabel("Iteration"); ax2.set_xlabel("X"); 
    ax1.set_ylabel("Cost"); ax2.set_ylabel("Y")
    
    ax1.plot(j_hist)
    ax2.plot(x_train, np.dot(X_scaled, model_w) + model_b, label="Predicted Value")

    for i in range(0, len(p_hist), 100):
        w, b = p_hist[i]
        ax3.plot(x_train, np.dot(X_scaled, w) + b)

    plt.show()
