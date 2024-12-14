from typing import List, Dict, Optional
from real_estate_toolkit.agent_based_model.houses import House
import statistics

class HousingMarket:
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses
    
    def get_house_by_id(self, house_id: int) -> House:
        """
        Retrieve specific house by ID.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        raise ValueError(f"House with ID {house_id} not found.")
    
    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate average house price, optionally filtered by bedrooms.
        """
        filtered_houses = self.houses
        if bedrooms is not None:
            filtered_houses = [house for house in self.houses if house.bedrooms == bedrooms]
        
        if not filtered_houses:
            raise ValueError("No houses available for this criteria.")
        
        prices = [house.price for house in filtered_houses]
        return statistics.mean(prices)
    
    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> List[House]:
        """
        Filter houses based on buyer requirements.
        """
        if segment == "FANCY":
            # Casas nuevas con alta calidad
            filtered_houses = [
                house for house in self.houses
                if house.is_new and house.quality_score >= 8 and house.price <= max_price
            ]
        elif segment == "OPTIMIZER":
            # Precio por pie cuadrado bajo
            filtered_houses = [
                house for house in self.houses
                if house.price / house.square_feet < 200 and house.price <= max_price
            ]
        elif segment == "AVERAGE":
            # Casas por debajo del precio promedio
            average_price = self.calculate_average_price()
            filtered_houses = [
                house for house in self.houses
                if house.price <= average_price and house.price <= max_price
            ]
        else:
            # Si el segmento no coincide, devuelve vacío
            filtered_houses = []

        return filtered_houses if filtered_houses else []
