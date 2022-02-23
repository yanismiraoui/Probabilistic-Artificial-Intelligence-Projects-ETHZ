import numpy as np


domain = np.array([[0, 6], [0, 6]])

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    v_dim_0 = np.all(x[:, 0] >= domain[0, 0]) and np.all(x[:, 0] <= domain[0, 1])
    v_dim_1 = np.all(x[:, 1] >= domain[1, 0]) and np.all(x[:, 0] <= domain[1, 1])

    return v_dim_0 and v_dim_1


def validate_input(x_test, n_points=None):
    """Check whether a point belongs to the domain and has the right shape."""
    x_test = np.array(x_test)
    x_test = np.atleast_2d(x_test)
    # Check domain size only
    if n_points is None:
        assert x_test.shape[1] == domain.shape[0], \
            f"The input must be a 2d array with {domain.shape[0]} columns. " \
            f"The input provided has {x_test.shape[1]} columns instead."
    # Check also number of points
    else:
        assert x_test.shape == (n_points, domain.shape[0]), \
            f"The input must be have shape {(n_points, domain.shape[0])}. " \
            f"The input has {x_test.shape} shape instead."
    assert check_in_domain(x_test), \
        f'The input must be within the domain, {x_test} provided instead.'
    return x_test
