"""split_movitation_column function to be called upon by the main script to
split the motivation column into multiple columns with binary values"""

import io
import pandas as pd
from pandas import DataFrame
import requests

# Constants:
SHEETURL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1sptWDnGOyRcEyCHFYhyC8Y_zXGGM5jMpePRVusoSkFs/"
    "edit?resourcekey=&gid=1530912831#gid=1530912831"
)

# format for CSV https://docs.google.com/spreadsheets/d/
# <SHEET_ID>/gviz/tq?tqx=
# out:csv&sheet=<SHEET_NAME>
SHEET_CSV_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1sptWDnGOyRcEyCHFYhyC8Y_zXGGM5jMpePRVusoSkFs/"
    "gviz/tq?tqx=out:csv&sheet=Altered/congregrated data"
)



def split_motivation_column(motivated_by_column: DataFrame = None) -> DataFrame:
    """
    splits the motivation column into multiple dataframe columns with binary
    values
    @throws relevant errors if the dataframe is empty, or more than one column
    is passed
    @param{df} @type{DataFrae}: the dataframe to split the motivation column
    @returns @type{DataFrame}: the new dataframe with the split columns
    """
    if motivated_by_column is None:
        df: DataFrame = get_data()
        motivated_by_column = pd.DataFrame(df["When playing games, I am most motivated by..."])

    if motivated_by_column.empty:
        raise ValueError("Dataframe is empty")

    if len(motivated_by_column.columns) > 1:
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
        new_df[motivation] = motivated_by_column[
            "When playing games, I am most motivated by..."
        ].apply(
            lambda x: 1 if motivation in x else 0
        )

    return new_df


def get_data() -> DataFrame:  # idealy we dont want to us Any, but for now
    """function to get the data from the google sheet
    raises: HTTPError: if the request fails (meaning url wrong or no inter)

    @returns: @type(DataFrame): the data from the google sheet"""

    response = requests.get(SHEET_CSV_URL)
    response.raise_for_status()  # Raise error if request fails
    df: DataFrame = pd.read_csv(io.StringIO(response.text))
    return df
