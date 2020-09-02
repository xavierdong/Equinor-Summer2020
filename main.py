
import matplotlib.pyplot as plt

from fetch import fetch_well_data
from model import *
from postprocessing import postprocess


def plot_graph(y_pred, y_validation):

    plt.yticks(np.arange(0, 1.3, 1))
    plt.xlabel('Binned Time')
    plt.ylabel('Interruption')
    plt.title("Actual vs Predicted Over Time w/ Binning")
    plt.plot(list(range(len(y_pred))), y_pred, "-b")
    plt.plot(list(range(len(y_validation))), y_validation, "-r")
    plt.legend(["Actual", "Predicted"])


if __name__ == '__main__':

    # update __sas_url if necessary
    # __sas_url = ('https://datavillagesa.blob.core.windows.net/volve?'
    #              'sv=2018-03-28&sr=c&sig=MgaLzfQcNK%2B%2FdMb3EyoF83U'
    #              '%2BvgKzQaxMo8O0ZbFhE6s%3D&se=2020-08-16T16%3A56%3A56Z&sp=rl')
    # fetch_well_data(__sas_url, ['15_9-F-5', '15_9-F-4'], 'fetch_results',
    #                 path_checker_csv="Drilling_Data_Path.csv", override=False)

    test_file = ('fetch_results/'
                 'WITSML_Realtime_drilling_data_Norway-StatoilHydro-15_$47$_9-F-5_1_log_1_1_1_combined.csv')
    validation_file = ('fetch_results/'
                       'WITSML_Realtime_drilling_data_Norway-StatoilHydro-15_$47$_9-F-4_2_log_1_3_1_combined.csv')

    rfc = model_random_forests(test_file, validation_file, 250, 2000, 'Seabed', True)
    x_validation_data, y_validation_data, label_dict_validation = \
        well_data_time_series_preprocessing(validation_file, 250, 2000, 'Seabed')

    x_validation_data = x_validation_data.reshape(np.shape(x_validation_data)[0], -1)
    y_pred_data = rfc.predict(x_validation_data)
    y_validation_data, y_pred_data = postprocess(y_validation_data, y_pred_data)

    plot_graph(y_pred_data, y_validation_data)
