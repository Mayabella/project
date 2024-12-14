from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Any
import csv

@dataclass
class DataLoader:
    """Class for loading and basic processing of real estate data."""
    data_path: Path
    
    from typing import Optional  # Para manejar valores None
    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """Load data from CSV file into a list of dictionaries."""
        try:
            with open(self.data_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
                reader = csv.DictReader(csvfile)
                data = [row for row in reader]

                # Limpieza inicial: Manejar valores vacíos y "NA"
                for row in data:
                    for key, value in row.items():
                        if value is None or value.strip().upper() in {"NA", ""}:
                            row[key] = None
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {self.data_path} does not exist.")
        except Exception as e:
            raise Exception(f"An error occurred while reading the CSV: {e}")


    
    def validate_columns(self, required_columns: List[str]) -> bool:
        """Validate that all required columns are present in the dataset."""
        data = self.load_data_from_csv()
        if not data:
            return False
        columns = set(data[0].keys())
        return all(column in columns for column in required_columns)
