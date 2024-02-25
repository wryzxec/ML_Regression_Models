from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_model_output(x, w, b):
    
    m = x.shape[0]
    f_wb = np.zeros(m)
    
    for i in range(m):
        f_wb[i] = w*x[i] + b
    
    return f_wb

def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = cost_sum/(2*m)

    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w*x[i] + b
        dj_dw += (f_wb - y[i])*x[i]
        dj_db += (f_wb - y[i])

    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):

    J_history = []
    p_history = []
    b = b_in
    w = w_in 

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        b = b - alpha*dj_db
        w = w - alpha*dj_dw

        J_history.append(cost_function(x, y, w , b))
        p_history.append([w,b])
            

    return w, b, J_history, p_history


if __name__ == "__main__":

    data_file_path = Path(__file__).parent / 'swedish_insurance_data.csv'
    df = pd.read_csv(data_file_path)

    x_train = df['X'].values
    y_train = df['Y'].values

    print(f"x_train: {x_train}")
    print(f"y_train: {y_train}")

    w_init = 0
    b_init = 0

    iterations = 100
    tmp_alpha = 1.0e-4

    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    print(f"w_final: {w_final}")
    print(f"b_final: {b_final}")
    print(f"cost_final: {compute_cost(x_train, y_train, w_final, b_final)}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(12,5))
    
    ax1.plot(J_hist[:100])

    ax2.scatter(x_train, y_train, marker='x', c='r')
    ax2.plot(x_train, w_final*x_train + b_final, c='b')

    for coefficients in p_hist:
        slope, intercept = coefficients
        y_values = slope * x_train + intercept
        ax3.plot(x_train, y_values)

    ax1.set_title("Cost vs. iteration(start)"); ax2.set_title("Model")
    ax1.set_ylabel('Cost')            ; ax2.set_ylabel('Number of claims')
    ax1.set_xlabel('iteration step')  ; ax2.set_xlabel('Total payment for claims in 1000s (Kr)')

    plt.show()

    