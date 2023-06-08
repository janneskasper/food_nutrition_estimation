import os
import pandas as pd
from fuzzywuzzy import fuzz, process 

class DensityDatabase():
    """Density Database searcher object. Food types are expected to be
    in column 1, food densities in column 2."""
    def __init__(self, db_path, sheet="Density DB"):
        """Load food density database from file or Google Sheets ID.

        Inputs:
            db_path: Path to database excel file (.xlsx) or Google Sheets ID.
        """
        print("Loading food density database ...")
        sheet = 'Density DB'
        if os.path.exists(db_path):
            # Read density database from excel file
            self.density_database = pd.read_excel(
                db_path, sheet_name=sheet, usecols=[0, 1])
        else:
            # Read density database from Google Sheets URL
            url = 'https://docs.google.com/spreadsheets/d/{0}/gviz/tq?tqx=out:csv&sheet={1}'.format(
                db_path, sheet)
            self.density_database = pd.read_csv(url, usecols=[0, 1],
                                                header=None)
        # Remove rows with NaN values
        self.density_database.dropna(inplace=True)

    def query(self, food):
        """Search for food density in database.

        Inputs:
            food: Food type to search for.

        Returns:
            db_entry_vals: Array containing the matched food type
            and its density.
        """
        try:
            # Search for matching food in database
            match = process.extractOne(food, self.density_database.values[:,0], score_cutoff=80)
            db_entry = (
                self.density_database.loc[
                self.density_database[
                self.density_database.columns[0]] == match[0]])
            return db_entry.values[0]
        except:
            return ['None', 1]
        
