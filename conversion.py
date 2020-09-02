from os import listdir, remove, makedirs
from os.path import isfile, join, exists, split, splitext
from tabula import read_pdf
from pathlib import Path

import re
import csv
import json
import os

import pandas as pd
from bs4 import BeautifulSoup as BSoup
import xml.etree.cElementTree as eTree


def merge_xml_to_csv(in_dir, output_name="output.csv", output_dir=None, del_temp=False, del_xml=False):
    """
    Converts all .xml files in in_dir to .csv and merges them into one .csv file

    Keyword arguments:
    in_dir -- path to input directory containing files
    output_name (optional) -- name of file as output
                           -- default: "output.csv"
    output_dir (optional) -- path to output directory
                          -- default: same as in_dir
    del_temp (optional) -- remove intermediate files
                        -- default: False
    del_xml (optional) -- remove .xml files used for conversion
                       -- default: False

    Return value -- tuple(str, list[str]):
    returns tuple with values: path to output, list of uids for dictionary
    """

    fn_list = [join(in_dir, f) for f in listdir(in_dir) if f.endswith(".xml") and isfile(join(in_dir, f))]

    if fn_list is None or len(fn_list) < 1:
        return None

    if output_dir is None:
        output_dir = in_dir

    # create output paths
    filename, file_extension = splitext(output_name)
    csv_out = join(output_dir, filename + '.csv')
    dict_out, readme_out = _fetch_dictionary_path(csv_out)

    # variable declarations
    all_dict = {}
    all_uid = []
    data = pd.DataFrame()

    csv_merge = open(csv_out, 'w')
    csv_list = [xml_to_csv(fn) for fn in fn_list]

    if del_xml:
        for f_xml in fn_list:
            remove(f_xml)

    # merging process
    for path, uid in csv_list:

        # merge csv files
        df = pd.read_csv(path)
        data = data.append(df, ignore_index=True)

        # merge the dictionaries
        my_dict = fetch_dictionary(path)[uid]
        all_dict[uid] = my_dict
        all_uid.append(uid)

        # remove temporary files if needed
        if del_temp:
            dict_file, readme_file = _fetch_dictionary_path(path)
            remove(path)
            remove(dict_file)
            remove(readme_file)

    csv_merge.close()

    # write dictionary to file
    all_uid = list(set(all_uid))

    with open(dict_out, 'w') as file:
        file.write(json.dumps(all_dict))
    data.to_csv(csv_out, index=False)
    write_readme_dictionary(readme_out, all_dict, all_uid)

    print("Merged csv files to", csv_out)
    return csv_out, all_uid


def xml_to_csv(filename, output=None):
    """
    Converts xml files to csv files

    Keyword arguments:
    filename -- Path to xml file
    output (optional) -- Path to destination with new filename
                      -- default: same path and name as filename

    Return value -- tuple(str, str):
    returns tuple with values: path to output, uid for log
    """

    if output is None:
        output = splitext(filename)[0] + '.csv'

    tree = eTree.parse(filename)

    # fetching the namespace from the group
    ns_m = re.match(r'{(.*)}', tree.getroot().tag)
    ns = {'mw': ns_m.group(1)}

    # initializing csv variables
    csv_file = open(output, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # fetching writing headers to file
    headers_obj = tree.find(".//mw:mnemonicList", namespaces=ns)
    headers = headers_obj.text.strip('\n').split(',')

    log_node = tree.find('.//mw:log', namespaces=ns)
    log_attr = log_node.attrib
    log_headers = list(log_attr.keys())

    others = []
    for node in log_node.iter():

        head_m = re.match(r'{.*}(.*)', node.tag).group(1)
        if head_m == 'logCurveInfo':
            break
        others.append((head_m, node.text))

    others = others[1:]
    others_val = [val for (header, val) in others]

    headers = log_headers + [header for (header, val) in others] + headers
    csv_writer.writerow(headers)

    # order of headers: <log headers>, <other headers before logCurveInfo>, <others>
    for data in tree.findall(".//mw:data", namespaces=ns):
        data_write = data.text.strip('\n').split(',')
        data_write = list(log_attr.values()) + others_val + data_write
        csv_writer.writerow(data_write)

    csv_file.close()

    # populating dictionary file
    uid = 'NA'
    if 'uid' in log_headers:
        uid = log_attr['uid']
    populate_dictionary(output, uid, tree.findall(".//mw:logCurveInfo", namespaces=ns))

    print("Finished conversion:", filename, "to", output)
    return output, uid


def populate_dictionary(filename, uid, lci_list):
    """
    Creates a dictionary and writes it as a json to a .txt file

    Keyword arguments:
    filename -- Path to location of .csv file
    uid -- uid attribute for the log
    lci_list -- list of logCurveInfo fetched from .xml file
    """

    output, readme_out = _fetch_dictionary_path(filename)

    # fetch list of dictionaries: one entry for each lci
    parsed_dict = {}
    for lci in lci_list:

        lci_dict = {}
        uid2 = lci.attrib['uid']

        count = 0
        for node in lci.iter():
            if count == 0:
                count += 1
                continue

            head_m = re.match(r'{.*}(.*)', node.tag).group(1)
            lci_dict[head_m] = node.text

        parsed_dict[uid2] = lci_dict

    # writing dictionary to file
    my_dict = {uid: parsed_dict}
    with open(output, 'w') as file:
        file.write(json.dumps(my_dict))

    write_readme_dictionary(readme_out, my_dict, [uid])


def write_readme_dictionary(filename, dictionary, uid_list):
    head_written = False
    csv_file = open(filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    headers = []

    for uid in uid_list:

        sub = dictionary[uid]
        for key, log_info in sub.items():

            if head_written:
                values = [log_info.get(k, '') for k in headers]
                csv_writer.writerow([uid] + values)
            else:
                headers = list(log_info.keys())
                csv_writer.writerow(["uid-main"] + headers)
                head_written = True
    csv_file.close()


def fetch_dictionary(filename):
    """
    Fetches the dictionary for the passed argument

    Keyword arguments:
    filename -- Path to the .csv file

    Return value -- dict:
    returns dictionary if file exists, None otherwise
    """

    f_path, o_path = _fetch_dictionary_path(filename)
    if isfile(f_path):
        file = open(f_path)
        dictionary = dict(json.load(file))
        file.close()
        return dictionary
    return None


def _fetch_dictionary_path(csv_path):
    _sub_dir = "dict"

    _base_dir, _f_name = split(csv_path)
    _sub_path = join(_base_dir, _sub_dir)

    if not exists(_sub_path):
        makedirs(_sub_path)

    _filename = splitext(_f_name)[0]
    return join(_sub_path, _filename + '.txt'), join(_sub_path, _filename + '_readme.csv')


def get_drill_log_tables(file_path_list, delete=False):
    """
    Converts list of paths to PDF files to one single Pandas dataframe of the
    operations data tables from each of the daily drilling log PDFS

    Keyword arguments:
    file_path_list -- list of paths to daily drilling report PDF files

    Returns -- list of dataframes that are the operations data tables from the
    daily drilling log PDFS
    """

    operations_df_list = []

    for pdf_path in file_path_list:
        df_list = read_pdf(pdf_path, pages=1)

        table_num = len(df_list)
        path = Path(pdf_path)

        # String manipulation on file to get Name and Date
        file_name = splitext(path.parts[-1])[0]  # Removes PDF ending
        # Regex for the datetime
        log_date = re.findall("([12]\d{3}_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01]))",
                              file_name)[0][0]  # re.findall returns a tuple in a list (need for element of tuple)

        # Need the name of the wellbore
        wellbore_name = file_name.replace("_" + log_date, "")

        # Replacing underscores with -'s to better fit real time drilling data
        log_date = log_date.replace("_", "-")
        wellbore_name = wellbore_name.replace("_", "-")

        # Need to add columns to dataframes with the name and date
        for i in range(table_num):
            curr_df = df_list[i]
            curr_cols = curr_df.columns
            # Get flag for if it goes through to continue to the next iteration of the big path
            log_date_list = [log_date] * curr_df.shape[0]
            wellbore_name_list = [wellbore_name] * curr_df.shape[0]

            if 'Start\rtime' in curr_cols:
                curr_df['log_date'] = log_date_list  # Fix repetative code
                curr_df['wellbore_name'] = wellbore_name_list
                operations_df_list.append(curr_df)  # We will want to continue
                break  # The breaks are just to make the code more efficient (less iterations)
            elif 'Start\rtime' in curr_df.iloc[0].tolist():
                new_df_cols = curr_df.iloc[0]
                curr_df['log_date'] = log_date_list
                curr_df['wellbore_name'] = wellbore_name_list
                curr_df = curr_df.drop(curr_df.index[0]).rename(columns=new_df_cols)
                operations_df_list.append(curr_df)
                break

    if delete:
        for path in file_path_list:
            remove(path)

    print("Number of operations dataframes sucessfully found: " + str(len(operations_df_list)))
    return operations_df_list


def convert_html_to_dataframe(html_path_list, delete=False):
    html_operations_df_list = []
    for html_path in html_path_list:
        if not isfile(html_path):
            print(str(html_path) + " is not a valid path.")
            continue

        curr_path = Path(html_path)

        # String manipulation on file to get Name and Date
        file_name = splitext(curr_path.parts[-1])[0]  # Removes PDF ending
        # Regex for the datetime (right now leaves the format with underscores)
        log_date = re.findall("([12]\d{3}_(0[1-9]|1[0-2])_(0[1-9]|[12]\d|3[01]))",
                              file_name)[0][0]  # re.findall returns a tuple in a list (need for element of tuple)

        # Need the name of the wellbore
        wellbore_name = file_name.replace("_" + log_date, "")

        # parsing html contents

        soup = BSoup(open(html_path), "html.parser")
        op_table = soup.find("table", {"id": "operationsInfoTable"})

        if op_table is None:
            print("Parsing HTMLs: cannot find operations table for", html_path)
            continue

        headers = op_table.find("thead").findAll("th")
        headers = [header.text for header in headers]

        rows = [[col.text for col in row.findAll("td")] for row in op_table.find("tbody").findAll("tr")]

        curr_df = pd.DataFrame(rows, columns=headers)
        # Maybe a conditional
        log_date_list = [log_date] * curr_df.shape[0]
        wellbore_name_list = [wellbore_name] * curr_df.shape[0]
        curr_df['log_date'] = log_date_list
        curr_df['wellbore_name'] = wellbore_name_list
        html_operations_df_list.append(curr_df)

    # for df in html_operations_df_list:
    #   print("(" + str(len(df.columns)) + ", " + str(df.columns) + ")")
    # print(html_operations_df_list[0].head())

    complete_df = pd.concat(html_operations_df_list, ignore_index=True)
    complete_df['log_date'] = complete_df['log_date'].map(lambda x: x.replace("_", "-"))
    complete_df['wellbore_name'] = complete_df['wellbore_name'].map(lambda x: x.replace("_", "-").replace("-", "_", 1))

    if delete:
        for path in html_path_list:
            os.remove(path)
    return complete_df