from pandasql import sqldf
import pandas as pd
from datetime import timedelta
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import smote_variants as sv

from scipy.stats import linregress


def filter_unused_columns(df):
    """
    Filters out the unused columns for the real-time drilling dataframe.
    :param df: real-time drilling dataframe
    :return: filtered dataframe
    """
    df = df[['TIME', 'ACTC', 'RPM', 'CHKP', 'SPPA', 'HKLD', 'ROP', 'SWOB', 'TQA', 'MWTI',
             'TVCA', 'TFLO', 'MDOA', 'CPPA', 'CFIA', 'nameWellbore', 'DMEA']]
    return df


def create_rolling_feature(df, window_size=10):
    """
    Combines the combined data with additional formation data

    Parameters
    ----------
    df : Pandas Dataframe
        WITSML Realtime drilling time from the Volve dataset taken from Azure
        or Combined Dataset

    window: rolling window size, 26 rows correspond to rougly 1 minute in real-time
    """

    def get_slope(array):
        y = np.array(array)
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return slope

    mean_feature = ['ROP', 'SWOB', 'TQA', 'RPM', 'HKLD', 'SPPA', 'CHKP', 'CPPA']
    range_feature = ['ROP', 'SWOB', 'TQA', 'RPM', 'HKLD', 'SPPA', 'CHKP', 'CPPA']
    slope_feature = ['ROP', 'SWOB', 'TQA', 'RPM', 'HKLD', 'SPPA', 'CHKP', 'CPPA']
    rolling_header_list = []

    for header in mean_feature:
        df[header + '_rolling_mean'] = df[header].rolling(window_size).mean()
        rolling_header_list.append(header + '_rolling_mean')

    # for header in range_feature:
    #     df[header + '_rolling_range'] = df[header].rolling(window_size).max() - df[header].rolling(window_size).min()
    #     rolling_header_list.append(header + '_rolling_range')

    for header in slope_feature:
        df[header + '_rolling_slope'] = df.groupby('nameWell')[header].rolling(window=window_size,
                                                                               min_periods=window_size).apply(get_slope,
                                                                                                              raw=False).reset_index(
            0, drop=True)
        rolling_header_list.append(header + '_rolling_slope')

    return df, rolling_header_list


def combine_time_and_pdf_data(real_time_drilling_data, daily_log_data):
    """
    Combines the real time drilling data with the operations table from
    the daily log drilling data

    Parameters
    ----------
    real_time_drilling_data : Pandas Dataframe
        WITSML Realtime drilling time from the Volve dataset taken from Azure
        Blob Storage converted into a Pandas Dataframe
    daily_log_data : Pandas Dataframe
        Operations data tables from the daily drilling reports in the Volve dataset
        converted intoa Pandas Dataframe from Azure Blob Storage
    """

    #Parse datetimes of TIME column into two seperate columns
    time = "TIME"
    # if time not in real_time_drilling_data:
    #     time = "Time"
    daily_log_data.rename(columns={"Start\rtime": "Start_time", "End\rtime": "End_time"})

    real_time_drilling_data['log_date'] = real_time_drilling_data[time].map(lambda x: x[:10])
    real_time_drilling_data['current_time'] = real_time_drilling_data[time].map(lambda x: x[11:-1])
    #Makes new wellbore_name column to be compatible with the daily_log_date
    real_time_drilling_data['wellbore_name'] = real_time_drilling_data['nameWellbore'].map(lambda x: (x[:8]).replace("/", "_"))

    print(daily_log_data.head())
    q = """
      SELECT *
      FROM real_time_drilling_data rtdd
      INNER JOIN daily_log_data dld 
        ON (rtdd.current_time BETWEEN
          dld.Start_time AND dld.End_time)
          AND (rtdd.log_date == dld.log_date)
          AND (rtdd.wellbore_name == dld.wellbore_name)
    """

    joined_df = sqldf(q)
    return joined_df

def combine_time_and_html_log_data(real_time_drilling_data, daily_log_data):
    """
    Combines the real time drilling data with the operations table from
    the daily log drilling data

    Parameters
    ----------
    real_time_drilling_data : Pandas Dataframe
        WITSML Realtime drilling time from the Volve dataset taken from Azure
        Blob Storage converted into a Pandas Dataframe
    daily_log_data : Pandas Dataframe
        Operations data tables from the daily drilling reports in the Volve dataset
        converted intoa Pandas Dataframe from Azure Blob Storage
    """

    #Parse datetimes of TIME column into two seperate columns
    time = 'TIME'
    # if time not in real_time_drilling_data:
    #     time = "Time"
    daily_log_data = daily_log_data.rename(columns={"Start time": "Start_time", "End time": "End_time"})

    real_time_drilling_data['original_time'] = real_time_drilling_data['TIME']
    real_time_drilling_data.astype({'TIME':'datetime64[ns]'})
    real_time_drilling_data['TIME'] = pd.to_datetime(real_time_drilling_data['TIME']) + timedelta(hours=8)

    real_time_drilling_data['log_date'] = real_time_drilling_data['TIME'].map(lambda x: x.date())
    real_time_drilling_data['current_time'] = real_time_drilling_data['TIME'].map(lambda x: x.strftime("%H:%M:%S"))
    real_time_drilling_data['wellbore_name'] = real_time_drilling_data['nameWellbore'].map(lambda x: (x[:8]).replace("/", "_"))

    q = """
      SELECT *
      FROM real_time_drilling_data rtdd
      INNER JOIN daily_log_data dld 
        ON (rtdd.current_time BETWEEN
          dld.Start_time AND dld.End_time)
          AND (rtdd.log_date == dld.log_date)
          AND (rtdd.wellbore_name == dld.wellbore_name)
    """

    print("Processing SQL querry")
    joined_df = sqldf(q)
    print("Finished join")
    return joined_df


def combine_drilling_and_formation_data(combined_data, formation_data):
    """
    Combines the combined data with additional formation data

    Parameters
    ----------
    combined_data : Pandas Dataframe
        WITSML Realtime drilling time from the Volve dataset taken from Azure
        Blob Storage converted into a Pandas Dataframe
        + drilling report log data

    formation_data : Pandas Dataframe
        formation dataframe with formation name and the top of formation depth
    """
    formation_data["Well name"] = formation_data["Well name"].str.lstrip('NO ')
    formation_data["Well name"] = formation_data["Well name"].str.replace("/", "_")
    formation_data = formation_data[["Well name", "Surface name", "MD", "TVD"]]
    formation_data = formation_data.rename(
        columns={"Well name": "wellbore_name", "Surface name": "Formation", "MD": "MD_Top", "TVD": "TVD_Top"})
    formation_data["MD_Bottom"] = np.nan
    formation_data["TVD_Bottom"] = np.nan

    fd_unique_md = formation_data["MD_Top"].unique()
    fd_unique_tvd = formation_data["TVD_Top"].unique()

    for i in formation_data.index:
        for j in range(len(fd_unique_md) - 1):
            if formation_data["MD_Top"][i] == fd_unique_md[j]:
                formation_data.loc[i, "MD_Bottom"] = fd_unique_md[j + 1]

    for i in formation_data.index:
        for j in range(len(fd_unique_tvd) - 1):
            if formation_data["TVD_Top"][i] == fd_unique_tvd[j]:
                formation_data.loc[i, "TVD_Bottom"] = fd_unique_tvd[j + 1]
    formation_data[["MD_Bottom", "TVD_Bottom"]] = formation_data[["MD_Bottom", "TVD_Bottom"]].fillna(10000)

    q = """
      SELECT *
      FROM combined_data cd
        INNER JOIN formation_data fd
          ON cd.DMEA >= fd.MD_Top
          AND cd.DMEA < fd.MD_Bottom
          AND cd.wellbore_name == fd.wellbore_name
    """

    joined_df = sqldf(q)
    return joined_df


def filter_by_run_number(joined_df, run_number):
    """
    Filters rows from the joined_df that are not the input run_number

    Parameters
    ----------
    joined_df : Pandas Dataframe
      Dataframe containing at least the realtime drilling data and the
      run data
    run_number : int
      Run number that will be used to filter the joined dataframe
    """

    joined_df = joined_df[joined_df['Run No.'] == str(run_number)]
    return joined_df


def group_wellbore_run_data(output_dir, input_wellbore_names):
    """
    Gets the wellbore run data from the volve Inventory file and creates a new
    dataframe that is groups the run numbers taking the minimum and maximum
    depth for the lower_bound and upper_bound columns
    Parameters
    ----------
    output_dir: string
        Output path returned after fetching the volve inventory data
    input_wellbore_names: list
      This is a list of the wellbore names that one wants to extract from the
      Volve Inventory file
    """

    wellbore_run_df_list = []
    volve_inventory = pd.ExcelFile(output_dir)

    volve_sheet_names = volve_inventory.sheet_names

    for input_name in input_wellbore_names:
        data = volve_inventory.parse(input_name)
        data.columns = data.iloc[0]
        data = data.drop(index=0, axis=1)
        # print(data.columns)

        run_data_dict = {'FOLDER': data['FOLDER'], 'Run No.': data['Run No.'], 'Interval': data['Interval']}
        run_data = pd.DataFrame(data=run_data_dict)

        run_data = run_data[run_data['Interval'] != 'TIME'] #Could also use regex

        run_data['lower_bound'] = run_data['Interval'].map(lambda x: str(x).split('-')[0])
        run_data['upper_bound'] = run_data['Interval'].map(
            lambda x: x if len(str(x).split("-")) == 1 else str(x).split("-")[1].replace(" m", ""))

        # Remove values if interval is not defined
        run_data = run_data.dropna(subset=['Interval', 'Run No.']) #need to worry about the TIME thing
        run_data = run_data[run_data['Run No.'].map(lambda run_num: len(
            str(run_num).split("-")) == 1)]  # Gets rid of the 1-4 b/c it is shown in other parts of the data
        run_data = run_data[
            run_data['FOLDER'] == 'LWD_EWL']  # This can be used to get rid of values for production logs if needed
        run_data['Run No.'] = run_data['Run No.'].map(
            lambda run_num: str(run_num))  # Fixes problems with strings vs. integers in data

        run_grouped_data = run_data.groupby(by='Run No.').agg(
            {'lower_bound': 'min', 'upper_bound': 'max'}).reset_index()
        input_name = input_name.replace(" ", "")
        run_grouped_data['wellbore_name'] = [input_name] * run_grouped_data.shape[0]

        wellbore_run_df_list.append(run_grouped_data)

    return wellbore_run_df_list

#RPM,CHKP,SPPA,HKLD,ROP,SWOB,TQA,MWTI,TVCA,TFLO,MDOA
def clean_dataframe(df):

    # null handling
    df['ACTC'] = df['ACTC'].fillna(method='ffill')
    df['ROP'] = df['ROP'].interpolate()
    df['TQA'] = df['TQA'].interpolate()
    df['SWOB'] = df['SWOB'].interpolate()
    df['RPM'] = df['RPM'].fillna(method='ffill')
    df['HKLD'] = df['HKLD'].interpolate()
    df['SPPA'] = df['SPPA'].interpolate()
    df['CHKP'] = df['CHKP'].interpolate()
    df['CPPA'] = df['CPPA'].interpolate()
    df['CFIA'] = df['CFIA'].interpolate()
    df['TFLO'] = df['TFLO'].interpolate()
    df['MWTI'] = df['MWTI'].fillna(df['MWTI'].mode())
    df['MDOA'] = df['MDOA'].fillna(df['MDOA'].mode())

    # remove other nulls
    df = df.dropna(axis=0, subset=['ACTC', 'ROP', 'TQA', 'SWOB', 'RPM', 'HKLD',
                                   'SPPA', 'CHKP', 'CPPA', 'CFIA', 'TFLO',
                                   'MWTI', 'MDOA'])

    # drop other improper values
    df = df[df['ROP'] > 0]
    df = df[df['TQA'] >= 0]
    df = df[df['SWOB'] >= 0]
    df = df[df['RPM'] >= 0]
    df = df[df['CFIA'] >= 0]
    df = df[df['TFLO'] >= 0]
    df = df[df['MWTI'] > 0]
    df = df[df['MDOA'] > 0]

    # TQA: remove >30
    df = df[df['TQA'] <= 30]

    return df

def well_data_time_series_preprocessing(well_data_dir, T, buffer_val, formation_name):
    """
    Gets the combined data and converts it to a format of distinct input data
    and labels
    """

    drilling_data = pd.read_csv(well_data_dir)
    print(np.shape(drilling_data))
    drilling_data = clean_dataframe(drilling_data)
    print(np.shape(drilling_data))
    print(drilling_data['Formation'].head())
    drilling_data = drilling_data[drilling_data['Formation'] == formation_name]
    print(np.shape(drilling_data))

    # drilling_data, header_list = create_rolling_feature(drilling_data, window_size = 62)

    drill_data_subset = drilling_data[['ACTC', 'RPM', 'CHKP', 'SPPA', 'HKLD', 'ROP', 'SWOB',
                                       'TQA', 'MWTI', 'TVCA', 'TFLO', 'MDOA', 'CPPA', 'CFIA', 'Main - Sub Activity']]

    # Get rid of ones with nothing for RPM
    drill_data_subset = drill_data_subset.dropna()
    # print(drill_data_subset.shape)
    drill_data_subset = drill_data_subset.reset_index(drop=True)

    scaler = StandardScaler()
    drill_data_subset[[
        'RPM', 'CHKP', 'SPPA', 'HKLD', 'ROP', 'SWOB',
        'TQA', 'MWTI', 'TVCA', 'TFLO', 'MDOA', 'CPPA', 'CFIA'
    ]] = scaler.fit_transform(
        drill_data_subset[
            ['RPM', 'CHKP', 'SPPA', 'HKLD', 'ROP', 'SWOB', 'TQA', 'MWTI', 'TVCA', 'TFLO', 'MDOA', 'CPPA', 'CFIA']])

    print(drill_data_subset.head())

    drill_data_subset['ACTC'] = drill_data_subset['ACTC'].astype('category')
    drill_data_subset['ACTC'] = drill_data_subset['ACTC'].cat.codes
    drill_data_subset['Main - Sub Activity'] = drill_data_subset['Main - Sub Activity'].map(
        lambda x: 'interruption' if 'interruption' in str(x) else 'Not interruption')

    series_labels = drill_data_subset[['Main - Sub Activity']].astype('category')
    label_dict = dict(enumerate(series_labels['Main - Sub Activity'].cat.categories))
    series_labels['Main - Sub Activity'] = series_labels['Main - Sub Activity'].cat.codes
    series_labels = series_labels.reset_index(drop=True)

    drill_data_subset['Main - Sub Activity'] = drill_data_subset['Main - Sub Activity'].astype('category').cat.codes

    # Drop the 'Main - Sub Activity'
    # drill_data_subset = drill_data_subset.drop(columns=['Main - Sub Activity'])

    # Converts the dataframes to numpy arrays
    series_data = drill_data_subset.values
    series_labels = series_labels.values

    # T = 500
    # buffer_val = 1000
    x_data = []
    y_data = []

    assert (np.shape(series_data)[0] == np.shape(series_labels)[0])

    for i in range(len(series_data) - T - buffer_val):
        x = series_data[i:i + T][:]
        # print(x)
        x_data.append(x)
        y = series_labels[i + T + buffer_val]  # I think this is all we have to do
        y_data.append(y)

    x_data = np.array(x_data)  # .reshape(-1, T, 1) # N x T x D (Need to reshape into 1 3d matrix )
    y_data = np.array(y_data)
    N = len(x_data)

    return x_data, y_data, label_dict


def oversampling_borderline_smote_1(x_data, y_data, prop):
    """

    :param x_data: the input minority label data for borderline_SMOTE1
    :param y_data: the output minority label data for boarderline_SMOTE1
    :param prop: float which is the proportion of the number of minority labels one wishes to oversample. Ex. prop=1.0
    will mean that if there are 100 minority labels the algorithm will create 100 new data points. If prop=0.5, then
    the algorithm will create 50 new data points
    :return: Returns the inputs x_data and y_data after the oversampling algorithm has been applied
    """
    x_data = x_data.reshape(np.shape(x_data)[0], -1)
    y_data = y_data.reshape(np.shape(y_data)[0], -1)

    oversampler = sv.Borderline_SMOTE1(proportion=prop)
    X_res, y_res = oversampler.sample(x_data, y_data.flatten())

    x_data = X_res
    y_data = y_res.reshape(-1, 1)

    return x_data, y_data


