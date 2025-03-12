"""split_movitation_column function to be called upon by the main script to
split the motivation column into multiple columns with binary values"""

import pandas as pd
from pandas import DataFrame


def split_motivation_column(df: DataFrame) -> DataFrame:
    """
    splits the motivation column into multiple dataframe columns with binary
    values
    @throws relevant errors if the dataframe is empty, or more than one column
    is passed
    @param{df} @type{DataFrae}: the dataframe to split the motivation column
    @returns @type{DataFrame}: the new dataframe with the split columns
    """
    if df.empty:
        raise ValueError("Dataframe is empty")

    if len(df.columns) > 1:
        raise ValueError("Only one column is allowed")

    motivation_types = [
        "Beating my competitors",
        "Mastering the game",
        "Earning the most points",
        "Working with a team",
        "Feeling immersed in the story/plot"
    ]

    # Create the new DataFrame
    new_df: DataFrame = pd.DataFrame()

    for motivation in motivation_types:
        new_df[motivation] = df[
            "When playing games, I am most motivated by..."
        ].apply(
            lambda x: 1 if motivation in x else 0
        )

    return new_df
