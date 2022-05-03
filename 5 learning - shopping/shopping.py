"""
AI to predict whether online shopping customers will complete a purchase.

$ python shopping.py shopping.csv
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
"""

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from datetime import datetime # for strptime to convert Feb to 2

TEST_SIZE = 0.4


def main():
    test = 0
    
    if test == 1:
        csvdata = "shopping.csv"
    else:
        # Check command-line arguments
        if len(sys.argv) != 2:
            sys.exit("Usage: python shopping.py data")
        csvdata = sys.argv[1]

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(csvdata)
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, "r") as csvfile:
        data = csv.DictReader(csvfile)
        evidence = []
        labels = []
        
        for line in data:
            # reset the linedata list for each iteration
            linedata = []
            
            # and off we go!
            # the csv.DictReader creates an ordered Dict, so we can retrieve the data with line[]
            linedata.append( int(line["Administrative"]) )
            linedata.append( float(line["Administrative_Duration"]) )
            linedata.append( int(line["Informational"]) )
            linedata.append( float(line["Informational_Duration"]) )
            linedata.append( int(line["ProductRelated"]) )
            linedata.append( float(line["ProductRelated_Duration"]) )
            linedata.append( float(line["BounceRates"]) )
            linedata.append( float(line["ExitRates"]) )
            linedata.append( float(line["PageValues"]) )
            linedata.append( float(line["SpecialDay"]) )
            
            # Month should be 0 for January, 1 for February, 2 for March, etc. up to 11 for December.
            # so get the three letters for the month, convert to the month number, and subtract 1 from that
            # we can use strptime for that: https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
            # note: the csv file has June and the other months are 3 letters, so we use [:3] to convert to a 3 letter string
            linedata.append( datetime.strptime(line["Month"][:3], "%b").month - 1 )
            
            linedata.append( int(line["OperatingSystems"]) )
            linedata.append( int(line["Browser"]) )
            linedata.append( int(line["Region"]) )
            linedata.append( int(line["TrafficType"]) )
            
            # do some calculations to get visitortype and weekend
            linedata.append( 1 if line["VisitorType"] == "Returning_Visitor" else 0 )
            linedata.append( 1 if line["Weekend"] == "TRUE" else 0 )
            
            # write the newly filled list to the evidence list
            evidence.append( linedata )
            # and lastly append the revenue data to the labels list
            labels.append( 1 if line["Revenue"] == "TRUE" else 0 )
            
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    model = KNeighborsClassifier(n_neighbors=1)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # initialise the variables at 0 so we can add to the in the for-loop    
    labelPos = 0
    truePos = 0
    labelNeg = 0
    trueNeg = 0
    
    # use zip to join the tuples together: https://www.w3schools.com/python/ref_func_zip.asp
    for label, pred in zip(labels, predictions):
        if label == 1:
            # to get the proportion, we need to know the total of positive labels
            labelPos += 1
            if pred == 1:
                # if label and pred are both 1, we add to true positive
                truePos += 1
        # if label and pred are both 0, we add to true negative
        elif label == 0:
            # to get the proportion, we need to know the total of negative labels
            labelNeg += 1
            if pred == 0:
                # if label and pred are both 0, we add to true positive
                trueNeg += 1
    
    # sensitivity and specificity are proportions of the totals of each type of label, so we divide
    sensitivity = float(truePos / labelPos)
    specificity = float(trueNeg / labelNeg)

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
