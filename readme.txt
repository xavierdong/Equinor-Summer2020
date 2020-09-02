Predicting Offshore Drilling Interruptions with Equinor's Opensource Volve Dataset
We used real time drilling data to make interruption predictions up to three hours into the future using a 
combination of the real time drilling data and other Volve dataset data including formation data and the 
real time drilling logs.



Get Combined Data Set.ipynb

Go to last cell in jupyter notebook
#Replace first input parameter with well names or list of well names that you want to fetch the combined dataset
#Replace second input parameter of fetch_well_data with path of where you want to write the combined dataset
#Replace third input parameter of fetch_well_data with path to Drilling_Data_Path.csv

Run the entire notebook to fetch combined dataset for specified list of wells
This is used to get the data from Equinor's Azure Blob Storage.
