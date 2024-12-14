from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any

@dataclass
class Descriptor:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None value per column.
        If columns = "all" then compute for all.
        Validate that column names are correct. If not make an exception.
        Return a dictionary with the key as the variable name and value as the ratio.
        """
        if columns == "all":
            columns = self.data[0].keys()
        else:
            for column in columns:
                if column not in list(self.data[0].keys()):
                    raise ValueError(f"Invalid column name: {column}")

        none_ratios = {}
        for column in columns:
            total = len(self.data)
            none_count = sum(1 for row in self.data if row.get(column) is None)
            none_ratios[column] = none_count / total if total > 0 else 0
        return none_ratios

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average value for numeric variables. Omit None values.
        If columns = "all" then compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable. If not make an exception.
        Return a dictionary with the key as the numeric variable name and value as the average
        """
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0][col], (int, float))]
        else:
            for column in columns:
                if column not in list(self.data[0].keys()) or not all(
                    isinstance(row.get(column), (int, float, type(None))) for row in self.data
                ):
                    raise ValueError(f"Invalid or non-numeric column: {column}")

        averages = {}
        for column in columns:
            values = [row[column] for row in self.data if row.get(column) is not None]
            averages[column] = sum(values) / len(values) if values else 0
        return averages

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median value for numeric variables. Omit None values.
        If columns = "all" then compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable. If not make an exception.
        Return a dictionary with the key as the numeric variable name and value as the median
        """
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0][col], (int, float))]
        else:
            for column in columns:
                if column not in list(self.data[0].keys()) or not all(
                    isinstance(row.get(column), (int, float, type(None))) for row in self.data
                ):
                    raise ValueError(f"Invalid or non-numeric column: {column}")

        medians = {}
        for column in columns:
            values = sorted(row[column] for row in self.data if row.get(column) is not None)
            n = len(values)
            if n == 0:
                medians[column] = 0
            elif n % 2 == 1:
                medians[column] = values[n // 2]
            else:
                medians[column] = (values[n // 2 - 1] + values[n // 2]) / 2
        return medians

    def percentile(self, columns: List[str] = "all", percentile: int = 50) -> Dict[str, float]:
        """Compute the percentile value for numeric variables. Omit None values.
        If columns = "all" then compute for all numeric ones.
        Validate that column names are correct and correspond to a numeric variable. If not make an exception.
        Return a dictionary with the key as the numeric variable name and value as the percentile value
        """
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be between 0 and 100.")
        
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0][col], (int, float))]
        else:
            for column in columns:
                if column not in list(self.data[0].keys()) or not all(
                    isinstance(row.get(column), (int, float, type(None))) for row in self.data
                ):
                    raise ValueError(f"Invalid or non-numeric column: {column}")

        percentiles = {}
        for column in columns:
            values = sorted(row[column] for row in self.data if row.get(column) is not None)
            n = len(values)
            if n == 0:
                percentiles[column] = 0
            else:
                k = (n - 1) * percentile / 100
                f = int(k)
                c = min(f + 1, n - 1)
                percentiles[column] = values[f] + (values[c] - values[f]) * (k - f)
        return percentiles

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """Compute the mode for variables. Omit None values.
        If columns = "all" then compute for all.
        Validate that column names are correct. If not make an exception.
        Return a dictionary with the key as the variable name and value as a tuple of the variable type and the mode.
        If the variable is categorical
        """
        if columns == "all":
            columns = self.data[0].keys()
        else:
            for column in columns:
                if column not in list(self.data[0].keys()):
                    raise ValueError(f"Invalid column name: {column}")

        modes = {}
        for column in columns:
            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                modes[column] = ("unknown", None)
                continue
            
            value_counts = {}
            for value in values:
                value_counts[value] = value_counts.get(value, 0) + 1
            
            mode_value = max(value_counts, key=value_counts.get)
            if isinstance(values[0], (int, float)):
                modes[column] = ("numeric", mode_value)
            else:
                modes[column] = ("categorical", mode_value)
        return modes

import numpy as np

@dataclass
class DescriptorNumpy:
    """Simplified class for describing data using basic NumPy functionalities."""
    data: List[Dict[str, Union[int, float, str, None]]]

    def none_ratio(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the ratio of None values per column."""
        if columns == "all":
            columns = self.data[0].keys()
        result = {}
        for column in columns:
            values = [row.get(column) for row in self.data]
            none_count = values.count(None)
            result[column] = none_count / len(values) if len(values) > 0 else 0.0
        return result

    def average(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the average for numeric columns."""
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0].get(col), (int, float))]
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            result[column] = np.mean(values) if values else 0.0
        return result

    def median(self, columns: List[str] = "all") -> Dict[str, float]:
        """Compute the median for numeric columns."""
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0].get(col), (int, float))]
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            result[column] = np.median(values) if values else 0.0
        return result

    def percentile(self, columns: List[str] = "all", percentile: float = 50) -> Dict[str, float]:
        """Compute the specified percentile for numeric columns."""
        if columns == "all":
            columns = [col for col in self.data[0] if isinstance(self.data[0].get(col), (int, float))]
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if isinstance(row.get(column), (int, float))]
            result[column] = np.percentile(values, percentile) if values else 0.0
        return result

    def type_and_mode(self, columns: List[str] = "all") -> Dict[str, Tuple[str, Any]]:
        """Determine the type and mode for each column."""
        if columns == "all":
            columns = self.data[0].keys()
        result = {}
        for column in columns:
            values = [row[column] for row in self.data if row.get(column) is not None]
            if not values:
                result[column] = ("unknown", None)
                continue
            value_counts = {val: values.count(val) for val in set(values)}
            mode_value = max(value_counts, key=value_counts.get)
            result[column] = (type(values[0]).__name__, mode_value)
        return result
