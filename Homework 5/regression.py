import numpy as np
import csv
from matplotlib import pyplot as plt
import random


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    open_file = open(filename)
    read_file = csv.reader(open_file)
    dataset = list(read_file)
    del dataset[0]
    for row in dataset:
        del row[0]
    return np.array(dataset, dtype = 'float')


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    stat_list = []
    for row in dataset:
        stat_list.append(row[col])
    print(len(stat_list))
    print("{:.2f}".format(sample_mean(stat_list)))
    print("{:.2f}".format(sample_stdev(stat_list)))

def sample_mean(dataset):
    sum = 0
    for element in dataset:
        sum += element
    return sum / len(dataset)

def sample_stdev(dataset):
    sum = 0
    average = sample_mean(dataset)
    for element in dataset:
        sum += (element - average)**2
    variance = sum/(len(dataset) - 1)
    return variance ** 0.5


def regression(dataset, cols, betas):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    for num, row in enumerate(dataset):
        sum = 0
        for index, coefficient in enumerate(betas):
            if index == 0:
                sum += coefficient
            else:
                sum += coefficient * row[cols[index-1]]
        mse += (sum - row[0]) ** 2
    return mse/(len(dataset))


def gradient_descent(dataset, cols, betas):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    for big_index in range(len(betas)):
        mse = 0
        for num, row in enumerate(dataset):
            sum = 0
            for index, coefficient in enumerate(betas):
                if index == 0:
                    sum += coefficient
                else:
                    sum += coefficient * row[cols[index-1]]
            if big_index == 0:
                mse += (sum - row[0])
            else:
                mse += (sum - row[0]) * row[cols[big_index - 1]]
        grads.append(2 * mse/ len(dataset))
    return np.array(grads)


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    old_betas = np.asarray(betas)
    new_betas = np.asarray([])
    for iteration in range(T):
        new_betas = np.subtract(old_betas, eta*gradient_descent(dataset, cols, old_betas))
        print(iteration + 1, "{:.2f}".format(regression(dataset, cols, new_betas)), *["{:.2f}".format(num) for num in new_betas])
        old_betas = new_betas


def compute_betas(dataset, cols):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    features = []
    features.append([1] * len(dataset))
    for num in cols:
        features.append(dataset[:,num])
    features = np.transpose(np.array(features))
    mse = None
    betas = np.linalg.inv(np.transpose(features) @ features)
    betas = betas @ np.transpose(features) @ dataset[:,0]
    return (regression(dataset, cols, betas), *betas)


def predict(dataset, cols, features):
    """
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    betas = compute_betas(dataset, cols)
    betas = betas[1:]
    result = 0
    for i, b in enumerate(betas):
        if i == 0:
            result = b
        else:
            result += b * features[i - 1]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear_model = []
    quadratic_model = []
    for val in X:
        z = np.random.normal(0, sigma)
        linear_model.append([betas[0] + betas[1] * val[0] + z, val[0]])
        z = np.random.normal(0, sigma)
        quadratic_model.append([alphas[0] + alphas[1] * val[0]**2 + z, val[0]])
    return np.array(linear_model), np.array(quadratic_model)


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # Generate dataset (n, 1)
    dataset = []
    for i in range(1000):
        dataset.append([random.randint(-100, 101)])

    # random betas and alphas
    list_range = list(range(-100, 0)) + list(range(1, 101))
    betas = np.random.choice(list_range, size=(2))
    alphas = np.random.choice(list_range, size=(2))
    # list of sigmas
    sigmas = []
    for i in range(-4, 6):
        sigmas.append(10 ** i)

    # list of linear and quadratic models for each sigma
    synth_datasets = []
    for s in sigmas:
        synth_datasets.append(synthetic_datasets(betas, alphas, dataset, s))
    mse_linear = []
    mse_quad = []
    for pair in synth_datasets:
        mse_linear.append(compute_betas(pair[0], cols=[0, 1])[0])
        mse_quad.append(compute_betas(pair[1], cols=[0, 1])[0])

    # plot points
    fig, ax = plt.subplots()
    ax.plot(sigmas, mse_linear, marker='o', label='linear')
    ax.plot(sigmas, mse_quad, marker='o', label='quadratic')

    # naming axes
    ax.set_xlabel("sigmas")
    ax.set_ylabel("MSEs")
    ax.set_yscale("log")
    ax.set_xscale("log")
    leg = ax.legend()
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
