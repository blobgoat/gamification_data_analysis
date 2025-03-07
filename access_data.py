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
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# for stubs, if it says ur missing them do  mypy --install-types
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

# format for CSV https://docs.google.com/spreadsheets/d/
# <SHEET_ID>/gviz/tq?tqx=
# out:csv&sheet=<SHEET_NAME>
SHEET_CSV_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1sptWDnGOyRcEyCHFYhyC8Y_zXGGM5jMpePRVusoSkFs/"
    "gviz/tq?tqx=out:csv&sheet=Altered/congregrated data"
)

# global variables for our current data purposes
Y_COLS = [
            "On a scale of 1 - 5 how successful do you feel you are in SEAL lab?",
            "On a scale of 1 - 5, how successful to do you feel your teammates are in SEAL lab?",
            "On a scale of 1 - 5, how successful do your peers think you are in SEAL lab?",
            "On whole, how would you rate your satisfaction in SEAL lab?"
            ]
#G:N
X_DEMO_COLS = [
            "What group are you primarily affiliated with in SEAL Life (shows up in SEAL clan life)?",
            "AGE (Congregated)",
            "Gender (CONGREGATED)",
            "How do you describe your sexual orientation?",	
            "Which categories best describe you?",
            "Do you have any chronic condition that substantially limit your life activities?",
            "If you have a disability, please indicate (if comfortable) the terms"
            "that best describe the condition(s)",	
            "Which economic class do you identify with?",
            "RELIGION (Congregated)"
            ]
# O:U
X_GAME_COLS = [
            "When playing games, I am most motivated by...",
            "I consider myself to be...",
            "When playing games, I consider myself to be...",
            "When playing games, I am generally...",
            "When playing games, I prefer to be...",
            "When playing games, I consider myself to be...",
            "When playing games, I generally..."
            ]
# V:AJ
X_SEAL_COLS = [
            "When I use the SEAL Sudoku Sheet Tools, I feel like I am playing a game. ",
            "I consider myself to be highly experienced with the SEAL Sheet Tools.",
            "I find the Sudoku Sheet Tools to be aesthetically pleasing.",
            "I think SEAL rank reflect my work and my team's work accurately.",
            "I think SEAL leaderboard reflect my work and my team's work accurately.",
            "I think SEAL YBR reflect my work and my team's work accurately.",
            "I think SEAL VisTools reflect my work and my team's work accurately.",
            "I think SEAL RaceTrack reflect my work and my team's work accurately.",
            "I think SEAL Battle Station reflect my work and my team's work accurately.",
            "I think SEAL Command Center reflect my work and my team's work accurately.",
            "I understand what my SEAL statistics mean (Lab HP, Sheet HP, YBR Gold Delta, and Training Score).",
            "I know exactly how my actions affect my lab statistics (Lab HP, Sheet HP, YBR Gold Delta, and Training Score).",
            "Using the Sudoku Sheet Tools helps me and my team stay on track.",	
            "Using the Sudoku Sheet Tools encourages me to take risks and challenge myself.",
            "Using the Sudoku Sheet Tools makes my work in SEAL more enjoyable."
            ]
#AO:AX
X_USABILITY_COLS = [
                    "I think that I would like to use this system frequently.",
                    "I found the system unnecessarily complex.",
                    "I thought the system was easy to use.",
                    "I think that I would need the support of a technical person to be able to use this system.",
                    "I found the various functions in this system were well integrated.",
                    "I thought there was too much inconsistency in this system.",
                    "I would imagine that most people would learn to use this system very quickly.",
                    "I found the system very cumbersome to use.",	
                    "I felt very confident using the system.",	
                    "I needed to learn a lot of things before I could get going with this system."
                ]
X_DROP_COLS = [
                "Timestamp",
                "Sudoku Sheet Tools are all the tools you use when actively engaging with SEAL life. "
                "Like Sudoku Clan Life, Dashboard, VisTools, RaceTrack, YBR, Kanban, Rank, Battle station, Venue, etc.",
                "What groups are you affiliated with in SEAL Life?",
                "Have you ever developed software as a programmer for Sudoku Sheet Tools?",
                "What is your current age?",
                "On scale of 1-10, how confusing were the questions on this survey?",
                "If you have any, we appreciate any additional feedback on the structure and questions within the survey",
                "SUS Overall score",	
                "Learnability subscore",
                "Usability subscore"
]

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

    #pre-processing
    x_data, y_data = split_xy(data, X_DROP_COLS)
    y1, y2, y3, y4 = [y_data[:, i] for i in y_data.shape[1]]

    # 70% train 20% validation 10% test
    x_data, y_data = split_xy(data, X_DROP_COLS)
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33) 

    # normalization
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    #feature selection

    # model

def split_xy(data, drop_cols):
    """function to split the x and y data into separate ndarrays based on
    a set of columns to be dropped

    @parameter: data @type(DataFrame): unprocessed data 
    @parameter: drop_cols @type(ndarray): array of names of columns to drop

    @returns: @type(ndarray): relevant x-values from data
    @returns: @type(ndarray): y-values from data"""

    x_data = data.drop(columns = drop_cols + Y_COLS)

    #handle X_SEAL_COLS: map disagree - agree as 1-5
    options_map = {'Strongly disagree': 1,
                    'Disagree': 2, 
                    'Neutral': 3,
                    'Agree': 4,
                    'Strongly agree': 5
                }
    x_data[X_SEAL_COLS] = x_data[X_SEAL_COLS].replace(options_map)

    # one-hot-encoding for categorical data (demographics, gaming)
    x_data = pd.get_dummies(x_data).to_numpy()

    y_data = data[Y_COLS].to_numpy()
    return x_data, y_data

def standardize(x_train, x_val, x_test):
    """function that standardizes data to normal gaussian distribution. Standardization
    calculation is applied only to the training data.

    @parameter: x_train @type(nd.array) processed x training data to be standardized 
    @parameter: x_val @type(nd.array) processed x val data to be standardized 
    @paremeter: x_test @type(nd.array) processed x-test data to be standardized
    """
    scaler = StandardScaler().fit(x_train) # only fit on training data
    x_train_stand = scaler.transform(x_train)
    x_val_stand = scaler.transform(x_val)
    x_test_stand = scaler.transform(x_test)

    return x_train_stand, x_val_stand, x_test_stand

def feature_selection(x_train, y_train):
    """function to apply LASSO regression on training data to select optimal
    features.
    
    @parameter: x_train @type(nd.array) standardized x_train data"""

    lasso_data = pd.DataFrame(columns=['l1_penalty','model','rmse_train','rmse_validation'])

    for l1 in l1_lambdas:
        lasso_model = Lasso(alpha = l1)
        lasso_model.fit(x_train, y_train)#model
        lasso_predict = lasso_model.predict(x_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, lasso_predict))
        
        
        lasso_val = lasso_model.predict(x_train)
        rmse_validation = np.sqrt(mean_squared_error(y_train, lasso_val))
        lasso_data.loc[len(lasso_data)] = [l1, lasso_model, rmse_train, rmse_validation]





# notice that in python things compile sequentially,
# so we have to have the main function at the end of the file
# and weird compiler stuff/stubs at bottom

# this is for compiling the code, so we can just run main
# needed so we can just hit the button and run the code
if __name__ == "__main__":
    main()
