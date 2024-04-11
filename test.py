#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import json
import subprocess
import time
import joblib

from score import score


def main():
    model = joblib.load("D:\\Cmi\\Applied ML\\assignment_3\\trained_model.joblib")
    
    unit_tests = [test_smoke_test, 
                  test_format_test, 
                  test_prediction_value,
                  test_threshold_0,
                  test_threshold_1,
                  test_obvious_spam, 
                  test_obvious_non_spam]
    
    result = {}
    for testfun in unit_tests:
        try:
            testfun(model)
            result[testfun.__name__] = True
        except AssertionError:
            result[testfun.__name__] = False
            
    for fname, passed in result.items():
        print(("PASSED:" if passed else "FAILED:"), fname)
    
    nPassed = sum(result.values())
    nFailed = len(result) - nPassed
    print("\n{} PASSED, {} FAILED".format(nPassed, nFailed))


def test_flask():
    try:
        test_flask_app()
        print("Integration test PASSED")
    except AssertionError:
        print("Integration test FAILED")


def test_flask_app():
    # Launch the Flask app in a separate process
    flask_process = subprocess.Popen(['python', 'app.py'])
   
    time.sleep(1)  # Wait for Flask app to start

    # Send a POST request to the /score endpoint
    data = {'text': 'Test text for prediction'}
    response = requests.post('http://localhost:5000/score', json=data)
    result = response.json()

    # Check if response contains 'prediction' and 'propensity' keys
    assert 'prediction' in result
    assert 'propensity' in result

    # Close the Flask app by terminating the process
    flask_process.terminate()
    flask_process.wait()
    

def test_smoke_test(model):
    text = "This is a smoke test"
    threshold = 0.5
    prediction, propensity_score = score(text, model, threshold)
    assert isinstance(prediction, bool)
    assert isinstance(propensity_score, float)


def test_format_test(model):
    text = "This is a format test"
    threshold = 0.5
    prediction, propensity_score = score(text, model, threshold)
    assert 0 <= propensity_score <= 1


def test_prediction_value(model):
    text = "This is a prediction value test"
    threshold = 0.5
    prediction, propensity_score = score(text, model, threshold)
    assert prediction in [0, 1]


def test_threshold_0(model):
    text = "This is a threshold 0 test"
    threshold = 0
    prediction, propensity_score = score(text, model, threshold)
    assert prediction == 1


def test_threshold_1(model):
    text = "This is a threshold 1 test"
    threshold = 1
    prediction, propensity_score = score(text, model, threshold)
    assert prediction == 0


def test_obvious_spam(model):
    text = "Buy now!"
    threshold = 0.5
    prediction, propensity_score = score(text, model, threshold)
    assert prediction == 1


def test_obvious_non_spam(model):
    text = "Thank you for your assistance"
    threshold = 0.5
    prediction, propensity_score = score(text, model, threshold)
    assert prediction == 0


if __name__ == "__main__":
    main()


# In[ ]:




