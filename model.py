import numpy as np
from sklearn.ensemble import RandomForestClassifier
from preprocessing import well_data_time_series_preprocessing, oversampling_borderline_smote_1
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix


def model_random_forests(combined_data_train_dir, combined_data_validation_dir, T, buffer_val, formation_name, oversampling=False, prop=1.0):
    """
    This function will train a random forest model for given set of parameters and output the confusion matrix for
    the newly trained model returning the trained model.
    :param combined_data_train_dir: String to the directory with the combined data that will be used to train the
    random forest model
    :param combined_data_validation_dir: String to the directory with the combined data (real time drilling data, daily
    drilling logs, and the formation data) that will be used as the validation data for the random forest model
    :param T: This integer value is the number of rows that will be used in as an input into the model. Ex. T = 250 will
    make the number of input features be T times the number of features.
    :param buffer_val: The integer number of rows that the model will predict ahead. Each row is 4-5 seconds. Ex.
    buffer_val = 2000 is predicting about 2 hours 40 minutes into the future.
    :param formation_name: String describing which formation that will be used for this model
    :parem oversampling: It is a boolean value. Its default is set to False. If it is set to True oversampling will be
    implemented
    :param prop: float which is the proportion of the number of minority labels one wishes to oversample. Ex. prop=1.0
    will mean that if there are 100 minority labels the algorithm will create 100 new data points. If prop=0.5, then
    the algorithm will create 50 new data points
    :return: The model that was trained by the random forest classifier
    """

    x_data, y_data, label_dict = well_data_time_series_preprocessing(combined_data_train_dir, T,
                                                                     buffer_val, formation_name)
    x_validation_data, y_validation_data, label_dict_validation = well_data_time_series_preprocessing(combined_data_validation_dir,
                                                                                                      T, buffer_val, formation_name)

    x_data = x_data.reshape(np.shape(x_data)[0], -1)
    y_data = y_data.reshape(np.shape(y_data)[0], -1)
    x_validation_data = x_validation_data.reshape(np.shape(x_validation_data)[0], -1)
    y_validation_data = y_validation_data.reshape(np.shape(y_validation_data)[0], -1)

    # Random Forest Model
    if oversampling:
        x_data, y_data = oversampling_borderline_smote_1(x_data, y_data, prop)

    #Random Forest Model
    rf = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=7, n_jobs=None,
                                bootstrap=True, oob_score=True, random_state=101)
    model = rf.fit(x_data, y_data)  # maybe I need to flatten everything

    # Outputting Results
    print('R^2 Training Score: {:.2f}'.format(rf.score(x_data, y_data)))
    print('OOB Score: {:.2f}'.format(rf.oob_score_))
    print('Validation Score: {:.2f}'.format(rf.score(x_validation_data, y_validation_data)))

    cm = get_confusion_matrix(model, x_validation_data, y_validation_data)
    print(cm)

    return model


def model_lstm(combined_data_train_dir, combined_data_validation_dir, T, buffer_val, formation_name, oversampling=False, prop=1.0):
    """
    This function will train a long short term-memory network for given set of parameters and output the confusion
    matrix for the newly trained model returning the trained model.
    :param combined_data_train_dir: String to the directory with the combined data that will be used to train the
    LSTM model
    :param combined_data_validation_dir: String to the directory with the combined data (real time drilling data, daily
    drilling logs, and the formation data) that will be used as the validation data for the LSTM model
    :param T: This integer value is the number of rows that will be used in as an input into the model. Ex. T = 250 will
    make the number of input features be T times the number of features.
    :param buffer_val: The integer number of rows that the model will predict ahead. Each row is 4-5 seconds. Ex.
    buffer_val = 2000 is predicting about 2 hours 40 minutes into the future.
    :param formation_name: String describing which formation that will be used for this model
    :parem oversampling: It is a boolean value. Its default is set to False. If it is set to True oversampling will be
    implemented
    :param prop: float which is the proportion of the number of minority labels one wishes to oversample. Ex. prop=1.0
    will mean that if there are 100 minority labels the algorithm will create 100 new data points. If prop=0.5, then
    the algorithm will create 50 new data points
    :return: The model that was trained by the LSTM classifier
    """
    x_data, y_data, label_dict = well_data_time_series_preprocessing(combined_data_train_dir, T,
                                                                     buffer_val, formation_name)
    x_validation_data, y_validation_data, label_dict_validation = well_data_time_series_preprocessing(
        combined_data_validation_dir,
        T, buffer_val, formation_name)

    if oversampling:
        x_data, y_data = oversampling_borderline_smote_1(x_data, y_data, prop)

    D = np.shape(x_data)[2]
    # Creating LSTM model for binary classification
    i = Input(shape=(T, D))
    x = LSTM(50)(i)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(i, x)

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.01), metrics='accuracy')

    x_data = x_data.reshape(np.shape(y_data)[0], T, D)
    y_data = y_data.reshape(-1, 1)

    r = model.fit(x_data, y_data, batch_size=512, epochs=3, validation_data=(x_validation_data, y_validation_data))

    cm = get_confusion_matrix(model, x_validation_data, y_validation_data)

    print(cm)

    return model


def get_confusion_matrix(model, x_validation_data, y_validation_data):
    """
    Returns the confusion matrix by comparing the predictions with the y validation data
    :param model: Model taken in after training on machine learning model
    :param x_validation_data: x data for validation on the model
    :param y_validation_data: y data for validation on the model
    :return: Confusion matrix based on predictions and actual validation data
    """

    y_pred = model.predict(x_validation_data)
    rounded_y_pred = np.around(y_pred, decimals=0)
    y_pred = rounded_y_pred.astype(int)

    cm = confusion_matrix(y_validation_data, y_pred)

    return cm

