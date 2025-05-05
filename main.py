"""module to access data from google sheet, for getting set up with pandas"""
# ^^^ the above is documentation style comments
# this is used to describe modules, but also important to put in functions
# and classes. This is the equivalent to /** */ in java and enables hovering
# to check annotations.
# these usually go inside or in the "middle" of the function
# but still before the code starts
import io
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import helper_functions.spreadsheet_specific_helpers as helper


# for stubs, if it says ur missing them do   mypy --install-types
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
    "1XcR48HZuC-mSFB-uKIxwPFhfRGVX7bWy100PhcLA8oM/"
    "edit?resourcekey=&gid=1780925762#gid=1780925762"
)

# format for CSV https://docs.google.com/spreadsheets/d/
# <SHEET_ID>/gviz/tq?tqx=
# out:csv&sheet=<SHEET_NAME>
SHEET_CSV_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1XcR48HZuC-mSFB-uKIxwPFhfRGVX7bWy100PhcLA8oM/"
    "gviz/tq?tqx=out:csv&sheet=Congregated Data"
)

# global variables for our current data purposes
Y_COLS = [
    "On a scale of 1 - 5 how successful do you feel you are in SEAL lab?",
    "On a scale of 1 - 5, how successful to do you feel your teammates are in SEAL lab?",
    "On a scale of 1 - 5, how successful do your peers think you are in SEAL lab?",
    "On whole, how would you rate your satisfaction in SEAL lab?"
]


# global variables for our current data purposes
Y_COLS = [
    "On a scale of 1 - 5 how successful do you feel you are in SEAL lab?",
    "On a scale of 1 - 5, how successful to do you feel your teammates are in SEAL lab?",
    "On a scale of 1 - 5, how successful do your peers think you are in SEAL lab?",
    "On whole, how would you rate your satisfaction in SEAL lab?"
]

X_DEMO_COLS = [
    "All SEAL group affiliations",
    "Age",
    "Gender",
    "Sexual orientation",
    "Race",
    "Chronic condition",
    "Condition description",
    "Economic class",
    "Religion"
]

X_PERSONALITY_COLS = [
    "Internal / External game motivation",
    "[Introverted - Extroverted]",
    "[Critical - Trusting]",
    "[Spontaneous - Conscientious]",
    "[Self-conscious - Even-tempered]",
    "[Prefer similarity - Am open to change]"
]
MOTIVATION_COLS = [
    "Beating my competitors",
    "Mastering the game",
    "Earning the most points",
    "Working with a team",
    "Feeling immersed in the story/plot"
]

X_SEAL_COLS = [
    "I feel like I am playing a game",
    "I consider myself to be highly experienced.",
    "Aesthetically pleasing.",
    "Rank reflects work accurately.",
    "Leaderboard reflects work accurately.",
    "YBR reflects work accurately.",
    "VisTools reflects work accurately.",
    "RaceTrack reflects work accurately.",
    "Battle Station reflects work accurately.",
    "Command Center reflects work accurately.",
    "I understand what my SEAL statistics mean.",
    "I know exactly how my actions affect my lab statistics",
    "Using the Sudoku Sheet Tools helps me and my team stay on track.",
    "Using the Sudoku Sheet Tools encourages me to take risks and challenge myself.",
    "Using the Sudoku Sheet Tools makes my work in SEAL more enjoyable."
]
# AO:AX
X_USABILITY_COLS = [
    'I think that I would like to use this system frequently',
    'I found the system unnecessarily complex',
    'I thought the system was easy to use',
    'I think that I would need the support of a technical person to be able to use this system',
    'I found the various functions in this system were well integrated',
    'I thought there was too much inconsistency in this system',
    'I would imagine that most people would learn to use this system very quickly',
    'I found the system very cumbersome to use',
    'I felt very confident using the system',
    'I needed to learn a lot of things before I could get going with this system.']

Y_COLS = ["Personal success",
          "Teammate success",
          "Peer success",
          "Satisfaction in SEAL"
]

X_DROP_COLS = ['All SEAL group affiliations', 'Game motivation']


def get_data() -> DataFrame:  # idealy we dont want to us Any, but for now
    """function to get the data from the google sheet
    raises: HTTPError: if the request fails (meaning url wrong or no inter)

    @returns: @type(DataFrame): the data from the google sheet"""

    response = requests.get(SHEET_CSV_URL)
    response.raise_for_status()  # Raise error if request fails
    df: DataFrame = pd.read_csv(io.StringIO(response.text))
    return df


def main():
    """main function to run the script"""
    # Code to be executed when the script is run directly
    data = get_data()

    # pre-processing
    x_data, y_data = split_xy(data)
    y1, y2, y3, y4 = [y_data.iloc[:, i] for i in range(y_data.shape[1])]
    x_nan = x_data.isna().sum().sum()
    y_nan = y_data.isna().sum().sum()
    print("NaN values", x_nan, y_nan)
    # NaN check

    x_data, y_data = split_xy(data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state = 12)

    # normalization
    x_train, x_test = standardize(x_train, x_test)

    # feature selection

    # model
    linear_regression(x_train, y_train)


def split_xy(data) -> Tuple[DataFrame, DataFrame]:
    """function to split the x and y data into separate ndarrays based on
    a set of columns to be dropped

    @parameter: data @type(DataFrame): rawdata

    @returns: @type(ndarray): relevant x-values from data
    @returns: @type(ndarray): y-values from data"""
    
    split_col: DataFrame = helper.split_motivation_column(data)
    x_data = data.drop(columns=X_DROP_COLS + Y_COLS)    # split motivation columns
    x_data = pd.concat([x_data, split_col], axis=1)

    # handle X_SEAL_COLS: map disagree - agree as 1-5
    options_map = {'Strongly disagree': 1,
                    'Disagree': 2,
                    'Neutral': 3,
                    'Agree': 4,
                    'Strongly agree': 5
                    }
    x_data[X_SEAL_COLS] = x_data[X_SEAL_COLS].replace(options_map)

    # Handle NaNs with data imputation of average
    x_data[X_PERSONALITY_COLS[1:]] = x_data[X_PERSONALITY_COLS[1:]].fillna(3)
    x_data[X_SEAL_COLS] = x_data[X_SEAL_COLS].fillna(3)
    x_data[X_USABILITY_COLS] = x_data[X_USABILITY_COLS].fillna(3)
    data[Y_COLS] = data[Y_COLS].fillna(3)

    # one-hot-encoding for categorical data (demographics, gaming)
    cat_col = x_data.select_dtypes(include=['object', 'category']).columns
    x_data = pd.get_dummies(x_data, columns = cat_col)
    y_data = data[Y_COLS]
    return x_data, y_data


def standardize(x_train, x_test) -> Tuple[DataFrame, DataFrame]:
    """function that standardizes data to normal gaussian distribution.
    Standardization calculation is applied only to the training data.

    @parameter: x_train @type(nd.array) processed x training data to be standardized
    @paremeter: x_test @type(nd.array) processed x-test data to be standardized
    """
    scaler = StandardScaler().fit(x_train)  # only fit on training data
    x_train_stand = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_stand = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    return x_train_stand, x_test_stand


def feature_selection(x_train, y_train):
    """function to apply LASSO regression on training data to select optimal
    features.

    @parameter: x_train @type(nd.array) standardized x_train data"""

    lasso_data = pd.DataFrame(
        columns=['l1_penalty', 'model', 'rmse_train', 'rmse_validation'])
    l1_lambdas = np.logspace(-4, 4, 100)
    for l1 in l1_lambdas:
        lasso_model = Lasso(alpha=l1)
        lasso_model.fit(x_train, y_train)  # model
        lasso_predict = lasso_model.predict(x_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, lasso_predict))

        lasso_val = lasso_model.predict(x_train)
        rmse_validation = np.sqrt(mean_squared_error(y_train, lasso_val))
        lasso_data.loc[len(lasso_data)] = [l1, lasso_model,
                                           rmse_train, rmse_validation]

    # inspect coefficients
    best_l1 = None
    rmse_test_lasso = None
    num_zero_coeffs_lasso = None
    indx = lasso_data['rmse_validation'].idxmin()
    best_l1 = lasso_data.loc[indx]['l1_penalty']
    best_mod = lasso_data.loc[indx]['model']
    num_zero_coeffs_lasso = np.count_nonzero(best_mod.coef_ == 0)
    print("Best L1", best_l1)
    print("num zero coef", num_zero_coeffs_lasso)

    # see minimized features
    all_features = x_train.columns
    zero_coef = []
    for feature, coef in zip(all_features, best_mod.coef_):
        if abs(coef) <= 10 ** -17:
            zero_coef.append(feature)
    print(zero_coef)

    return best_l1


def linear_regression(x_train, y_train, x_val, y_val):
    models = []
    train_rmse = []
    val_rmse = []
    for i in range(y_train.shape[1]):
        y_t = y_train.iloc[:, i]
        y_v = y_val.iloc[:, i]
        model = LinearRegression().fit(x_train, y_t)
        predict_t = model.predict(x_train)
        t_rmse = np.sqrt(mean_squared_error(y_t, predict_t))
        predict_v = model.predict(x_val)
        v_rmse = np.sqrt(mean_squared_error(y_v, predict_v))

        train_rmse.append(t_rmse)
        val_rmse.append(v_rmse)
        models.append(model)
    print(x_train.columns)
    linear_visualization(models, x_train.columns, y_train.columns)
    return models, train_rmse, val_rmse


def linear_visualization(models, features, y_cols):
    """function to visualize the linear regression model coefficients"""
    for i, model in enumerate(models):
        coef = model.coef_

        coef_list = pd.DataFrame({"Feature": features, "Coef": coef})
        coef_list["abs"] = coef_list["Coef"].abs()
        top = coef_list.nlargest(10, "abs")
        print(top["Feature"], top["Coef"])

        plt.figure(figsize=(10, 5))
        plt.barh(top["Feature"], top["Coef"])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Features")
        plt.ylabel("Coef values")
        plt.title(f"{y_cols[i]}")
        plt.show()


# notice that in python things compile sequentially,
# so we have to have the main function at the end of the file
# and weird compiler stuff/stubs at bottom
# this is for compiling the code, so we can just run main
# needed so we can just hit the button and run the code
if __name__ == "__main__":
    main()
