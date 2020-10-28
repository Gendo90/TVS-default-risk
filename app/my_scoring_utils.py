from sklearn.metrics import make_scorer

def equi_loans_val(y_true, y_pred):
    valid_predictions = 0
    invalid_predictions = 0

    for i in range(0, len(y_pred)):
        if (y_pred[i] == 1):
            if (y_true.array[i] == 1):
                valid_predictions += 1
            else:
                invalid_predictions += 1

    return valid_predictions*5-invalid_predictions


my_scorer = make_scorer(equi_loans_val, greater_is_better=True)
