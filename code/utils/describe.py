def compare_datasets(train_df, test_df, target_col):
    """
    Compare training and test datasets.

    Numeric variables:
        - Mean

    Categorical variables:
        - Percentage of observations in each category

    Returns:
        A DataFrame with:
        - train
        - test
        - percent_difference
    """

    import pandas as pd
    import numpy as np

    train = train_df.drop(columns=[target_col], errors="ignore")
    test = test_df.drop(columns=[target_col], errors="ignore")

    output = {}

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
        if (
            pd.api.types.is_numeric_dtype(train_series)
            and pd.api.types.is_numeric_dtype(test_series)
        ):
            output[column] = {
                "train": train_series.mean(),
                "test": test_series.mean()
            }

        # Categorical variables
        else:
            categories = pd.Index(
                train_series.dropna().unique().tolist()
                + test_series.dropna().unique().tolist()
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

                row_name = f"{column} = {category}"

                output[row_name] = {
                    "train": train_percentage,
                    "test": test_percentage
                }

    result = pd.DataFrame.from_dict(output, orient="index")

    result["percent_difference"] = np.where(
        result["train"] != 0,
        ((result["test"] - result["train"]) / result["train"]) * 100,
        np.nan
    )

    result.columns = ["train", "test", "percent_difference"]

    return result

