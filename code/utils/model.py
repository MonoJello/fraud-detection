import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def cv_auc_logit(X, y, features, cv):
    """Return mean cross-validated AUC for a statsmodels logistic model."""



    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = sm.add_constant(
            X.iloc[train_idx][features],
            has_constant="add"
        )
        X_test = sm.add_constant(
            X.iloc[test_idx][features],
            has_constant="add"
        )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        try:
            model = sm.Logit(y_train, X_train).fit(
                disp=False,
                maxiter=200
            )

            predictions = model.predict(X_test)
            aucs.append(roc_auc_score(y_test, predictions))

        except Exception:
            # Handles singular matrices or failed model fits
            aucs.append(np.nan)

    return np.nanmean(aucs)

def forward_select_auc(X, y, min_improvement=0.001, random_state=42):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )



    remaining = list(X.columns)
    selected = []

    best_auc = 0.5
    history = []

    while remaining:
        results = []

        for candidate in remaining:
            features = selected + [candidate]

            auc = cv_auc_logit(
                X=X,
                y=y,
                features=features,
                cv=cv
            )

            results.append((auc, candidate))

        candidate_auc, best_candidate = max(results)

        if candidate_auc > best_auc + min_improvement:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_auc = candidate_auc

            history.append({
                "added": best_candidate,
                "features": selected.copy(),
                "cv_auc": best_auc
            })

            print(
                f"Added {best_candidate:30s} "
                f"CV AUC = {best_auc:.4f}"
            )
        else:
            break

    return selected, pd.DataFrame(history)

def plot_logit_marginal_effect(
    model,
    data,
    variable,
    selected_features,
    grid_size=100,
    ax=None
):
    """
    Plot the marginal effect of `variable`, holding all other selected
    features at their sample means.
    """

    if variable not in selected_features:
        raise ValueError(
            f"{variable!r} must be included in selected_features."
        )

    missing = [
        feature for feature in selected_features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"These selected features are not in data: {missing}"
        )

    # All selected features except the variable being plotted
    variables_to_hold = [
        feature for feature in selected_features
        if feature != variable
    ]

    # Values over which to evaluate the target variable
    x_values = np.linspace(
        data[variable].min(),
        data[variable].max(),
        grid_size
    )

    # Create prediction data
    new_data = pd.DataFrame({
        variable: x_values
    })

    # Hold all remaining selected features at their means
    for feature in variables_to_hold:
        new_data[feature] = data[feature].mean()

    # Add any model terms not explicitly listed, if needed
    exog_names = model.model.exog_names

    for name in exog_names:
        if name != "const" and name not in new_data.columns:
            if name in data.columns:
                new_data[name] = data[name].mean()

    # Match the exact order used when fitting the model
    exog = new_data.reindex(
        columns=exog_names,
        fill_value=1
    )

    # Predicted probabilities
    probabilities = model.predict(exog)

    # Logistic marginal effect:
    # dP/dx = p(1-p) * beta
    beta = model.params[variable]

    marginal_effects = (
        probabilities * (1 - probabilities) * beta
    )

    results = pd.DataFrame({
        variable: x_values,
        "predicted_probability": probabilities,
        "marginal_effect": marginal_effects
    })

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.plot(
        results[variable],
        results["marginal_effect"],
        linewidth=2
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(variable)
    ax.set_ylabel("Marginal effect")
    ax.set_title(
        f"Marginal effect of {variable}\n"
        "Other selected features held at their means"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return results

