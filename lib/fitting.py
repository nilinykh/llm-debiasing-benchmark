from pathlib import Path
from ppi_py import ppi_logistic_pointestimate, ppi_ols_pointestimate
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pandas as pd
import tempfile

##########
# Linear #
##########

def linear_fit(X, Y):
    """
    Fit a linear regression to the data and return the coefficients
    """
    linear = LinearRegression(
        fit_intercept = True, # also fit the intercept
    )
    linear.fit(X, Y)
    coeffs = np.insert(linear.coef_, 0, linear.intercept_)
    return coeffs

def linear_fit_ppi(X, Y_true, Y_pred, selected):

    # Add intercept
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate([ones, X], axis=1)
    selected_mask = np.zeros(X.shape[0], dtype=bool)
    selected_mask[selected] = True

    return ppi_ols_pointestimate(
        X = X[selected_mask],
        Y = Y_true[selected_mask],
        Yhat = Y_pred[selected_mask],
        X_unlabeled = X[~selected_mask],
        Yhat_unlabeled = Y_pred[~selected_mask],
    )

def linear_fit_dsl(X, Y_true, Y_pred, selected):
    """
    Fit a logistic regression to the data with predicted and expert annotations
    using the DSL from the R package
    """

    # Set Y_true to None for non-selected samples
    not_selected = set(range(Y_true.shape[0])) - set(selected)
    not_selected = list(not_selected)
    Y_true_sel = Y_true.copy()
    Y_true_sel[not_selected] = None # selected expert annotations

    # Put the data into a dataframe
    data = pd.DataFrame({
        "c0": X[:,0],
        "c1": X[:,1],
        "c2": X[:,2],
        "c3": X[:,3],
        "Y": Y_true_sel,
        "Y_hat": Y_pred,
    })

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp:

        # Create temporary files
        data_file = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"

        # Write data to file
        data.to_csv(data_file, index=False)

        # R script that reads the data file, runs the DSL algorithm and
        # writes the coefficients to the result file
        ro.r(f"""
            sink("/dev/null")
            library("dsl")
            data <- read.csv("{data_file}")
            out <- dsl(
                model = "lm",
                formula = Y ~ c0 + c1 + c2 + c3,
                predicted_var = "Y",
                prediction = "Y_hat",
                data = data,
                seed = Sys.time()
            )
            write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
            sink()
        """)

        # Read coefficients from the result file
        coeffs = np.array(pd.read_csv(coeff_file)).squeeze()

    return coeffs


############
# Logistic #
############

def logit_fit(X, Y):
    """
    Fit a logistic regression to the data and return the coefficients
    """
    logreg = LogisticRegression(
        fit_intercept = True, # also fit the intercept
        penalty = None, # no regularisation (default in R)
        max_iter = 1000,
    )
    logreg.fit(X, Y)
    return np.insert(logreg.coef_, 0, logreg.intercept_)

def logit_fit_ppi(X, Y_true, Y_pred, selected):

    # Add intercept
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate([ones, X], axis=1)
    selected_mask = np.zeros(X.shape[0], dtype=bool)
    selected_mask[selected] = True

    return ppi_logistic_pointestimate(
        X = X[selected_mask],
        Y = Y_true[selected_mask],
        Yhat = Y_pred[selected_mask],
        X_unlabeled = X[~selected_mask],
        Yhat_unlabeled = Y_pred[~selected_mask],
        lam = 1,
    )

def logit_fit_dsl(X, Y_true, Y_pred, selected):
    """
    Fit a logistic regression to the data with predicted and expert annotations
    using the DSL from the R package
    """

    import rpy2.robjects as ro

    # Set Y_true to None for non-selected samples
    not_selected = set(range(Y_true.shape[0])) - set(selected)
    not_selected = list(not_selected)
    Y_true_sel = Y_true.copy()
    Y_true_sel[not_selected] = None # selected expert annotations

    # Put the data into a dataframe
    num_features = X.shape[1]
    features = [f"c{i}" for i in range(num_features)]
    data = {feature: X[:,i] for i, feature in enumerate(features)}
    data["Y"] = Y_true_sel
    data["Y_hat"] = Y_pred
    data = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp:

        data_file = Path(tmp) / "data.csv"
        coeff_file = Path(tmp) / "coeff.csv"

        data.to_csv(data_file, index=False)

        # R script that reads the data file, runs the DSL algorithm and
        # writes the coefficients to the result file
        ro.r(f"""
            sink("/dev/null")
            library("dsl")
            data <- read.csv("{data_file}")
            out <- dsl(
                model = "logit",
                formula = Y ~ {" + ".join(features)},
                predicted_var = "Y",
                prediction = "Y_hat",
                data = data,
                seed = Sys.time()
            )
            write.csv(out$coefficients, "{coeff_file}", row.names=FALSE)
            sink()
        """)

        # Read coefficients from the result file
        coeffs = np.array(pd.read_csv(coeff_file)).squeeze()

    return coeffs

# ###########
# # Own DSL #
# ###########
#
# def logistic_moment(beta, X, Y):
#
#     assert len(beta.shape) == 1
#     assert len(X.shape) == 2
#     assert len(Y.shape) == 1
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == beta.shape[0]
#
#     moment = (Y - expit(X @ beta))[:, np.newaxis] * X
#     return moment
#
# def fit_dsl_own(X, Y_true, Y_pred, selected, seed):
#
#     assert len(X.shape) == 2
#     assert len(Y_true.shape) == 1
#     assert len(Y_pred.shape) == 1
#     assert X.shape[0] == Y_pred.shape[0]
#     assert Y_true.shape[0] == len(selected)
#
#     Y_hat = np.zeros_like(Y_pred)
#
#     kf = KFold(n_splits = 10, shuffle = True, random_state = seed)
#     for train_idx, test_idx in kf.split(np.zeros(X.shape[0])):
#
#         idx = [i for i in train_idx if i in selected]
#         X_train = np.column_stack((X[idx,:], Y_pred[idx]))
#         idx = [np.where(selected == i)[0] for i in idx]
#         Y_train = np.ravel(Y_true[idx])
#
#         rf = RandomForestRegressor(
#             n_estimators = 2000,
#             max_depth = None,
#             random_state = seed,
#             min_samples_leaf = 5,
#         )
#         rf.fit(X_train, Y_train)
#
#         X_test = np.column_stack((X[test_idx, :], Y_pred[test_idx]))
#         Y_hat[test_idx] = rf.predict(X_test)
#
#     XX = np.insert(X, 0, 1, axis = 1)
#
#     def loss(beta):
#         m_pred = logistic_moment(beta, XX, Y_hat)
#         m_true = logistic_moment(beta, XX[selected,:], Y_true)
#         pi = len(selected) / XX.shape[0]
#         moments = m_pred
#         moments[selected] = m_pred[selected] - (m_pred[selected] - m_true) / pi
#         return np.linalg.norm(moments)
#
#     beta0 = np.zeros(XX.shape[1])
#     result = minimize(
#         fun = loss,
#         x0 = beta0,
#         options = {"maxiter": 1000},
#         method = "L-BFGS-B",
#         tol = 1e-7,
#     )
#
#     return result.x
