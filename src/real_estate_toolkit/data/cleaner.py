from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Cleaner:
    """Class for cleaning real estate data."""
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """
        Rename the columns with best practices (e.g., snake_case very descriptive name).
        """
        if not self.data:
            raise ValueError("Data is empty. Cannot rename columns.")

        # Extract column names from the first row of the data
        original_columns = self.data[0].keys()

        # Generate new column names with best practices
        def transform_column_name(name: str) -> str:
            """
            Transform a column name into snake_case:
            - Strip leading/trailing spaces.
            - Replace spaces, dashes, and dots with underscores.
            - Remove non-alphanumeric characters except underscores.
            - Convert to lowercase.
            """
            name = name.strip()  # Remove leading/trailing spaces
            name = name.replace(" ", "_").replace("-", "_").replace(".", "_")  # Normalize separators
            name = "".join(char for char in name if char.isalnum() or char == "_")  # Keep only valid characters
            return name.lower()  # Convert to lowercase

        seen = {}
        def resolve_conflict(name: str) -> str:
            if name in seen:
                seen[name] += 1
                return f"{name}_{seen[name]}"
            seen[name] = 0
            return name

        rename_map = {col: transform_column_name(col) for col in original_columns}

        # Rename columns in all rows
        for row in self.data:
            for old_name, new_name in rename_map.items():
                row[new_name] = row.pop(old_name)

    
    
    def na_to_none(self) -> List[Dict[str, Any]]:
        """Replace NA to None in all values with NA in the dictionary. """
        # Iterate through each row and column, replacing "NA" with None.
        for row in self.data:
            for key, value in row.items():
                if isinstance(value, str) and value.strip().upper() == "NA":
                    row[key] = None

        return self.data
    
    def clean_numeric_column(self, column_name: str) -> None:
        """" 
        Ensure that a specific column contains only numeric values.
        Replace invalid or non-numeric values with None.
        """
        if not self.data:
            raise ValueError("Data is empty. Cannot clean columns.")
                
        for row in self.data:
            value = row.get(column_name, None)
            try:
                # Intenta convertir el valor a float
                row[column_name] = float(value) if value is not None else None
            except (ValueError, TypeError):
                # Si el valor no es convertible a número, reemplaza con None
                row[column_name] = None

