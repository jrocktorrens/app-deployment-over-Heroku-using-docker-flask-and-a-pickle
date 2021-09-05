import requests
import pandas as pd
import numpy as np
import random

random.seed(42)
RANDOM_ROWS_TEST = random.sample(range(1, 100), 10)  # return a list of random rows to test predictions
COLUMNS = ['is_male', 'num_inters', 'late_on_payment', 'age', 'years_in_contract']


def main():
    """
    this function test the output from the inference server to the test predictions.
    """
    testing_int_array_from_train = np.loadtxt('pred.csv').astype(int)
    test_pred = [testing_int_array_from_train[idx] for idx in RANDOM_ROWS_TEST]
    inf_output_pred = rest_api()    # return a list of predictions from inference server
    assert (test_pred != inf_output_pred) == 0, 'predictions didnt match y_test'
    print('Success, we have a prediction!')


def rest_api():
    """
    this function sends data to inference server a nd get a prediction based on the data.
    the function return a list of predictions.
    """
    inference_output_list = []
    X_test = pd.read_csv('X_test.csv', usecols=COLUMNS)
    for idx in RANDOM_ROWS_TEST:
        sampled_xtest = pd.DataFrame(X_test.iloc[idx, :])
        curr_line_para = {col: [sampled_xtest.loc[col]] for col in COLUMNS}  # parameters for request
        single_response = requests.get('http://127.0.0.1:5000/predict_churn', params=curr_line_para)
        single_inference_pred = int(single_response.text.rsplit('</h1>')[1])
        inference_output_list.append(single_inference_pred)
    return inference_output_list


if __name__ == '__main__':
    main()
