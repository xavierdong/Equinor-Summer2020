import os
import re
import csv

from azure.storage.blob import ContainerClient
from pathlib import Path
import pandas as pd

from conversion import merge_xml_to_csv, convert_html_to_dataframe
from preprocessing import combine_time_and_html_log_data, combine_drilling_and_formation_data, \
    filter_unused_columns


def fetch_well_data(sas_url, well_names, out_dir, path_checker_csv=None, override=False):
    """
    Downloads all well-related files to specified file
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param well_names: list of well names to fetch data for
    :type well_names: list[str]
    :param out_dir: output directory to download files to
    :type out_dir: str
    :param path_checker_csv: a path to a csv containing a restricted list of useful Daily drilling data paths
    :type path_checker_csv: str
    :param override: override files if files for same well is found
    :type override: bool
    """
    sas_dict = fetch_valid_paths(sas_url, path_checker_csv)

    # create directory if doesn't exist
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # separating info into formation and well categories
    formation_path = sas_dict['formation']
    sas_dict = sas_dict['wells']
    path_tuple = [(w, w_paths) for w, w_paths in sas_dict.items() if w in well_names]

    # fetch formation file
    print('Fetching formation files')
    formation_file = fetch_formation(sas_url, out_dir, formation_path)
    formation_dir = split_formation(formation_file)

    # fetch well-specific files
    for w, w_paths in path_tuple:

        print("Fetching files for", w)
        out_paths = fetch_all_drilling_data(sas_url, out_dir, sas_dict, w, override)
        html_path_list = fetch_html_daily_drilling_logs(sas_url, out_dir, sas_dict, well_name=w)

        print("Finished fetching drilling and HTML files")
        log_df_concatenated = convert_html_to_dataframe(html_path_list, delete=True)
        formation_df_path = os.path.join(formation_dir, __get_well_pick_file_name(w))
        formation_df = pd.read_csv(formation_df_path)

        for out_path in out_paths:
            combine_files(out_path, log_df_concatenated, formation_df, override)


def fetch_valid_paths(sas_url, path_checker_csv):
    """
    Fetches all valid blob paths for the Azure Blob Storage for the following categories:
        1. Realtime drilling data
        2. Daily drilling reports - HTML
        3. Formation data
    NOTE: a potential cause of errors will be if the paths change in the Equinor VOLVE database, in which case,
    solution would be to update the paths in this function

    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param path_checker_csv: a path to a csv containing a restricted list of useful Daily drilling data paths
    :type path_checker_csv: str
    :return: a dictionary containing all valid paths for realtime drilling data, daily drilling reports,
    and formation data
    :rtype: dict[
        'formation': str,
        'well': dict[
            str: dict[
                'drill': list[str],
                'html_reports': list[str]
            ]
        ]
    ]
    """

    # update paths if file structure changes
    __real_time_drilling_path = 'WITSML Realtime drilling data'
    __daily_log_html_path = 'Well_technical_data/Daily Drilling Report - HTML Version/'
    __formation_data_path = 'Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat'

    container = ContainerClient.from_container_url(container_url=sas_url)

    drilling_paths = {}
    html_reports_paths = {}
    formation_path = ''
    valid_drill_subs = __get_valid_drill_sub(path_checker_csv)

    print("Fetch: retrieving list of blobs")
    blob_list = list(container.list_blobs())

    print("Fetch: filtering list of blobs")
    for blob in blob_list:

        name = blob.name
        if __real_time_drilling_path in name \
                and 'log' in name \
                and os.path.splitext(name)[1] == '.xml' \
                and (valid_drill_subs is None or __is_valid_drill(name, valid_drill_subs)):

            # extract well code
            well = __get_well_name_from_drill_sub(name)
            # append to data
            well_drills = drilling_paths.get(well, [])
            well_drills.append(name)

            drilling_paths[well] = well_drills
        elif __daily_log_html_path in name:

            # extract well code
            well = __get_well_name_from_report_blob(name, ".html")
            # append to data
            well_reports = html_reports_paths.get(well, [])
            well_reports.append(name)
            html_reports_paths[well] = well_reports

        elif __formation_data_path in name:
            formation_path = name

    return {
        'formation': formation_path,
        'wells': {
            well_name: {
                "drill": drilling_paths[str(well_name)],
                "html_reports": html_reports_paths.get(str(well_name), [])
            } for well_name in drilling_paths.keys()
        }
    }


def fetch_formation(sas_url, out_dir, formation_path):
    """
    Downloads the formation data file
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param out_dir: output directory to download files to
    :type out_dir: str
    :param formation_path: VOLVE path to the formation data file
    :type formation_path: str
    :return: path to downloaded file
    """

    __file_name = 'all_formations.dat'
    out_path = os.path.join(out_dir, __file_name)

    container = ContainerClient.from_container_url(container_url=sas_url)
    blob_client = container.get_blob_client(formation_path)
    download = blob_client.download_blob()

    with open(out_path, 'w') as f:
        f.write(download.readall().decode("utf-8"))

    return out_path


def split_formation(in_path, out_path=None):
    """
    Separates formation data into separate files on a well-name basis
    :param in_path: path to the formation data file
    :type in_path: str
    :param out_path: path to output
    :type out_path: str
    :return: path to folder containing the separated files
    """

    __default_out_dir = 'well_picks'

    if out_path is None:
        out_path = os.path.join(os.path.split(in_path)[0], __default_out_dir)
        # if directory doesn't create, create it
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

    with open(in_path) as f:
        lines = f.readlines()

    well_name = None
    header = None
    col_nums = None

    out_file = None
    csv_writer = None

    for line in lines:

        if not line.strip():
            col_nums = None

        if 'Well NO' in line:
            well_name = re.search('Well NO (.*)', line).group(1).strip()
        elif 'Well name' in line:
            header = line
        elif '-' in line and header is not None and col_nums is None:

            if col_nums is None:
                if out_file is not None:
                    out_file.close()

                out_file = open(os.path.join(out_path, __get_well_pick_file_name(well_name)), 'w')
                csv_writer = csv.writer(out_file)
                col_nums = __parse_columns(line)
                csv_writer.writerow(__parse_well_pick_line(header, col_nums))

        elif col_nums is not None and len(line.strip()) > 0 and line.count('-') <= 10 and 'Well NO' not in line:
            csv_writer.writerow(__parse_well_pick_line(line, col_nums))

    if out_file is not None:
        out_file.close()

    return out_path


def __get_combined_filename(csv_out):
    filename, file_extension = os.path.splitext(csv_out)
    csv_out = filename + '_combined.csv'
    return csv_out


def combine_files(out_path, log_df_concatenated, formation_df, override=False):
    """
    Combining the well-related data into one
    Output file: <data_file_name>_combined.csv
    :param out_path: path to output directory
    :type out_path: str
    :param log_df_concatenated: Pandas DataFrame containing the daily drilling log data
    :param formation_df: Pandas DataFrame containing formation data
    :param override: override files if files for same well is found
    :type override: bool
    """
    csv_out = __get_combined_filename(out_path)
    if os.path.isfile(csv_out) and not override:
        print(f'Failed to combine for {csv_out}: file already exists')
        return
    elif not os.path.isfile(out_path):
        print(f'Failed to combine for {csv_out}: {out_path} does not exists')
        return

    drill_df = pd.read_csv(out_path)
    drill_df = filter_unused_columns(drill_df)

    print(f'Combining for {csv_out}')

    com_df = combine_time_and_html_log_data(drill_df, log_df_concatenated)
    com_df = combine_drilling_and_formation_data(com_df, formation_df)
    os.remove(out_path)

    com_df.to_csv(csv_out, index=False)


def fetch_all_drilling_data(sas_url, out_dir, sas_dict, well_name, override=False):
    """
    fetches all realtime drilling data for one well
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param out_dir: pre-existing output directory
    :type out_dir: str
    :param sas_dict: dictionary fetched from fetch_sas_path function
    :type sas_dict: dict[str, dict[list[str]]]
    :param well_name: name of well to fetch data for
    :type well_name: str
    :param override: override old file if same file name found
    :type override: bool
    :return: paths to the output files
    :rtype: list[str]
    """

    data = sas_dict.get(well_name, None)
    if data is None:
        print("Cannot find paths for", well_name)
        return

    sub_folders = {}
    for path in data.get("drill"):
        key = __get_drilling_file_name(Path(path))
        sub = sub_folders.get(key, [])
        sub.append(path)
        sub_folders[key] = sub

    out_paths = []

    for sub, paths in sub_folders.items():
        out_path, dict_uids = fetch_sub_drilling_data(sas_url, out_dir, sub, paths, override)
        out_paths.append(out_path)

    return out_paths


def fetch_sub_drilling_data(sas_url, out_dir, filename, sub_paths, override=False):
    """Fetch all drilling data for one sub folder and convert into a single .csv file
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param out_dir: pre-existing file output directory
    :type out_dir: str
    :param filename: output file name (must be generated from __get_file_name)
    :type filename: str
    :param sub_paths: list of paths to the individual blobs in the sub folder
    :type sub_paths: list[str]
    :param override: override old file if same file name found
    :type override: bool
    """

    out_path = os.path.join(out_dir, filename)
    if (os.path.isfile(out_path) or os.path.isfile(__get_combined_filename(out_path))) and not override:
        print("Initiate fetch:", filename, "failed = file already exists")
        return out_path, None

    print("Initiating fetch: drilling data")
    container = ContainerClient.from_container_url(container_url=sas_url)

    print("Fetch: downloading and converting files in directory =", out_dir)
    c_name = __get_drilling_file_name(Path(sub_paths[0]))
    for path in sub_paths:
        blob_client = container.get_blob_client(path)
        download = blob_client.download_blob()

        path = Path(path)
        with open(os.path.join(out_dir, path.parts[-1]), 'w') as f:
            f.write(download.readall().decode("utf-8"))

    if os.path.isfile(os.path.join(out_dir, c_name)) and override:
        print("Fetch: preparing to override", c_name)
    out_path, dict_uids = merge_xml_to_csv(out_dir, output_name=c_name, del_temp=True, del_xml=True)

    print("Fetch: finished")
    return out_path, dict_uids


def fetch_pdf_daily_drilling_logs(sas_url, out_dir, sas_dict, well_name=None):
    """
    Takes PDF daily drilling well logs from the Volve dataset
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param out_dir: Path where the PDF files will be downloaded
    :type out_dir: str
    :param sas_dict: dictionary returned from get_valid_paths function
    :type sas_dict: dictionary
    :param well_name: name of well to fetch data for
    :type well_name: str
    :return: list of paths to the daily drilling logs
    :rtype: list[str]
    """

    container = ContainerClient.from_container_url(container_url=sas_url)

    data = sas_dict.get(well_name, None)
    if data is None:
        print("Cannot find paths for", well_name)
        return

    filtered_daily_log_list = data.get("pdf_reports")
    print("Number of daily logs taken: " + str(len(filtered_daily_log_list)))
    out_paths = []

    for path in filtered_daily_log_list:
        path = Path(path)
        blob_client = container.get_blob_client(str(path))
        download = blob_client.download_blob()
        out_path = os.path.join(out_dir, path.parts[-1])
        out_paths.append(out_path)
        with open(out_path, 'wb') as f:
            f.write(download.readall())

    print("Finished Getting Azure PDF Data")
    return out_paths


def fetch_html_daily_drilling_logs(sas_url, out_dir, sas_dict, well_name=None):
    """
    Takes HTML daily drilling well logs from the VOLVE dataset
    :param sas_url: an SAS URl to access the VOLVE Azure database
    :type sas_url: str
    :param out_dir: Path where the PDF files will be downloaded
    :type out_dir: str
    :param sas_dict: dictionary returned from get_valid_paths function
    :type sas_dict: dictionary
    :param well_name: name of well to fetch data for
    :type well_name: str
    :return: list of paths to the daily drilling logs
    :rtype: list[str]
    """

    container = ContainerClient.from_container_url(container_url=sas_url)

    path_list = []

    data = sas_dict.get(well_name, None)
    if data is None:
        print("Cannot find paths for", well_name)
        return

    filtered_daily_log_list = data.get("html_reports")

    print("Number of daily logs taken: " + str(len(filtered_daily_log_list)))

    for path in filtered_daily_log_list:
        path = Path(path)
        blob_client = container.get_blob_client(str(path))
        download = blob_client.download_blob()
        path_list.append(os.path.join(out_dir, path.parts[-1]))
        with open(os.path.join(out_dir, path.parts[-1]), 'wb') as f:
            f.write(download.readall())

    print("Finished Getting Azure HTML Data")

    return path_list


def __filter_wellbore_name(log_element):
    """
    Parses wellbore name from a daily drilling log file name
    :param log_element: daily drilling log file name
    :type log_element: str
    :return: wellbore name
    :rtype: str
    """
    new_path = Path(log_element)
    file_name = os.path.splitext(new_path.parts[-1])[0]
    log_date = re.findall(r"([12]\d{3}_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01]))",
                          file_name)[0][0]  # re.findall returns a tuple in a list (need for element of tuple)

    # #Need the name of the wellbore
    wellbore_name_blob = file_name.replace("_" + log_date, "")
    wellbore_name_blob = wellbore_name_blob.replace("_", "-")

    return wellbore_name_blob


def __is_valid_drill(drill_path, valid_subs):
    """
    checks to see if the drilling data path is valid given a list of valid sub folders
    :param drill_path: path to drilling data file
    :type drill_path: str
    :param valid_subs: list of valid subfolders
    :type valid_subs: list[str]
    :return: whether the path is valid
    :rtype: bool
    """
    path = Path(drill_path)

    for valid in valid_subs:
        if len(path.parts) > len(valid) + 1 and path.parts[1:len(valid) + 1] == valid:
            return True
    return False


def __get_valid_drill_sub(filename):
    """
    Parse a list of valid drilling data folders with respect to the file
    :param filename: path to file containing a list of valid folders for each well
    :type filename: str
    :return: list of folder paths
    :rtype: list[str]
    """

    if filename is None:
        return None

    sub_folders = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        header = False
        for row in reader:

            if not header:
                header = True
                continue

            if header:
                for col in row:
                    if col:
                        col = col.replace("\\", "/").replace("\'", "").replace("\"", "")
                        sub_folders.append(Path(col).parts)
                break
    return sub_folders


def __get_well_pick_file_name(well_name):
    """
    Standardizes well_pick file names
    :param well_name: name of well
    :type well_name: str
    :return: file name
    :rtype: str
    """
    return '_'.join(['well_pick', well_name.replace('/', '_') + ".csv"])


def __get_well_name_from_drill_sub(name):
    """
    Parses well name from a real-time drilling data folder
    :param name: path to folder
    :type name: str
    :return: name of well
    :rtype: str
    """

    target = [part for part in Path(name).parts if "_$47$_" in part][0].replace("_$47$_", "_")
    well = re.match("[ A-Za-z/-]+(.+)", target, flags=re.DOTALL).group(1).replace(" ", "")
    well = well[:-1] if well[-1].isalpha() else well
    return well


def __get_well_name_from_report_blob(name, file_extension):
    """
    parse well name from daily drilling log file name
    :param name: path to file
    :type name: str
    :param file_extension: file extension of the log file
    :type file_extension: str
    :return: name of well
    :rtype: str
    """

    target = Path(name).parts[-1]
    log_date = re.findall(
        r"([12]\d{3}_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01]))",
        target
    )[0][0]  # re.findall returns a tuple in a list (need for element of tuple)
    well = target.replace("_" + log_date, "").replace(file_extension, "")
    well = well.rsplit("_", well.count("_") - 1)
    well = '-'.join(well[:-1] if well[-1].isalpha() else well)
    return well


def __get_drilling_file_name(path, full_path=True):
    """
    standardizes real-time drilling data download file name
    :param path: path to real-time drilling data file in Volve
    :type path: Path
    :param full_path: whether the path includes the file name
    :type full_path: bool
    :return: path to download file
    :rtype: str
    """
    return ('_'.join(path.parts[0:-1] if full_path else path.parts) + ".csv").replace(" ", "_")


def __parse_columns(line):
    """
    parse the number of characters per column for the well picks .dat file
    :param line: a '-'-only line
    :type line: str
    :return: list of number of characters for each column
    :rtype: list[int]
    """
    columns = line.strip().split()
    col_nums = [col.count('-') + 1 for col in columns]
    return col_nums


def __parse_well_pick_line(line_str, col_nums):
    """
    split line on a column basis
    :param line_str: line to parse
    :type line_str: str
    :param col_nums: number of characters per column
    :type col_nums: list[int]
    :return: list containing values on column-basis
    :rtype: list[str]
    """
    line_str = line_str.strip()
    col_vals = []

    start_i = 0
    for i in range(len(col_nums)):
        col_vals.append(line_str[start_i: min(start_i + col_nums[i], len(line_str))].strip())
        start_i += col_nums[i]
    return col_vals
