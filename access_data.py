"""module to access data from google sheet, for getting set up with pandas"""
# ^^^ the above is documentation style comments
# this is used to describe modules, but also important to put in functions
# and classes. This is the equivalent to /** */ in java and enables hovering
# to check annotations.
# these usually go inside or in the "middle" of the function
# but still before the code starts

import typing
import requests
import io
import pandas as pd
# use pylint, mypy and pep8 extensions as linters (you might have to install
# with pip inaddition to the python extension)
# this is what types look like in python, we are going to use this to type
# hint our functions
# and also to help us keep track of variables

# global variables from javascript!
# but notice the lenght of the coding lines, this is a python convention.
# get the data at
SHEETURL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1sptWDnGOyRcEyCHFYhyC8Y_zXGGM5jMpePRVusoSkFs/"
    "edit?resourcekey=&gid=1530912831#gid=1530912831"
)

# format for CSV https://docs.google.com/spreadsheets/d/<SHEET_ID>/gviz/tq?tqx=out:csv&sheet=<SHEET_NAME>
SHEET_CSV_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1sptWDnGOyRcEyCHFYhyC8Y_zXGGM5jMpePRVusoSkFs/"
    "gviz/tq?tqx=out:csv&sheet="
)


def get_data() -> typing.Any:  # idealy we dont want to us Any, but for now
    """function to get the data from the google sheet"""

    response = requests.get(SHEET_CSV_URL)
    response.raise_for_status()  # Raise error if request fails
    df = pd.read_csv(SHEETURL)
    return df


def main():
    """main function to run the script"""
    # Code to be executed when the script is run directly

    # print('hello world')
    print(get_data())
    # Call other functions or perform operations here


# notice that in python things compile sequentially,
# so we have to have the main function at the end of the file
# and weird compiler stuff/stubs at bottom

# this is for compiling the code, so we can just run main
# needed so we can just hit the button and run the code
if __name__ == "__main__":
    main()
