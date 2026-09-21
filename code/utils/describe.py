def compare_datasets(train_df, test_df, target_col):
    """
    Compare training and test datasets.

    Numeric variables:
        - Mean
        - Absolute difference
        - Relative percent difference
        - Train/test ratio
        - Whether the mean changed sign

    Categorical variables:
        - Percentage of observations in each category
        - Difference in percentage points
        - Relative percent difference
        - Whether the category percentage changed from zero

    Returns:
        A DataFrame with:
        - train
        - test
        - difference
        - percent_difference
        - ratio
        - sign_changed
    """

    import numpy as np
    import pandas as pd

    train = train_df.drop(columns=[target_col], errors="ignore")
    test = test_df.drop(columns=[target_col], errors="ignore")

    rows = []

    columns = list(dict.fromkeys(
        train.columns.tolist() + test.columns.tolist()
    ))

    for column in columns:
        train_series = (
            train[column]
            if column in train.columns
            else pd.Series(dtype="object")
        )

        test_series = (
            test[column]
            if column in test.columns
            else pd.Series(dtype="object")
        )

        # Numeric variables
        is_numeric = (
            column in train.columns
            and column in test.columns
            and pd.api.types.is_numeric_dtype(train_series)
            and pd.api.types.is_numeric_dtype(test_series)
        )

        if is_numeric:
            train_value = train_series.mean()
            test_value = test_series.mean()

            difference = test_value - train_value

            percent_difference = (
                difference / train_value * 100
                if train_value != 0
                else np.nan
            )

            ratio = (
                test_value / train_value
                if train_value != 0
                else np.nan
            )

            sign_changed = (
                np.sign(train_value) != np.sign(test_value)
                if train_value != 0 and test_value != 0
                else False
            )

            rows.append({
                "attribute": column,
                "train": train_value,
                "test": test_value,
                "difference": difference,
                "percent_difference": percent_difference,
                "ratio": ratio,
                "sign_changed": sign_changed
            })

        # Categorical variables
        else:
            categories = pd.Index(
                train_series.dropna().tolist()
                + test_series.dropna().tolist()
            ).unique()

            for category in categories:
                train_percentage = (
                    train_series.eq(category).mean() * 100
                    if len(train_series) > 0
                    else np.nan
                )

                test_percentage = (
                    test_series.eq(category).mean() * 100
                    if len(test_series) > 0
                    else np.nan
                )

                difference = test_percentage - train_percentage

                percent_difference = (
                    difference / train_percentage * 100
                    if train_percentage != 0
                    else np.nan
                )

                ratio = (
                    test_percentage / train_percentage
                    if train_percentage != 0
                    else np.nan
                )

                sign_changed = (
                    np.sign(train_percentage)
                    != np.sign(test_percentage)
                    if train_percentage != 0 and test_percentage != 0
                    else False
                )

                rows.append({
                    "attribute": f"{column} = {category}",
                    "train": train_percentage,
                    "test": test_percentage,
                    "difference": difference,
                    "percent_difference": percent_difference,
                    "ratio": ratio,
                    "sign_changed": sign_changed
                })

    result = pd.DataFrame(rows).set_index("attribute")

    return result

