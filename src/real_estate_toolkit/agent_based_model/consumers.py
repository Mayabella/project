from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket

class Segment(Enum):
    FANCY = auto()  # Prefers new construction with the highest house scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Prefers houses below the average market price

@dataclass
class Consumer:
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3  # 30% of annual income reserved for savings
    interest_rate: float = 0.05  # Annual interest rate for savings
    
    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time.
        
        Formula: Future Value = P * (1 + r)^t
        Where:
        - P is the principal (savings at the beginning of the year)
        - r is the interest rate (annually)
        - t is the time in years
        """
        # Annual savings based on the saving_rate
        annual_savings = self.annual_income * self.saving_rate
        
        # Compound interest formula for savings growth over time
        for _ in range(years):
            self.savings += annual_savings  # Add savings for the year
            self.savings *= (1 + self.interest_rate)  # Apply interest

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house from the housing market.
        
        Matching logic:
        - Check if the consumer has enough savings for a down payment
        - House must match the consumer's segment preferences
        - House price should be affordable based on savings, income, and down payment requirements
        """
        if self.house:
            print(f"Consumer {self.id} already owns a house.")
            return
        
        # Get average house price in the market
        average_price = housing_market.calculate_average_price()
        
        # Filter houses based on consumer's segment preferences
        if self.segment == Segment.FANCY:
            # Look for new houses with the highest quality score
            suitable_houses = [house for house in housing_market.houses if house.is_new and house.quality_score == 10]
        elif self.segment == Segment.OPTIMIZER:
            # Focus on price per square foot value (under monthly salary)
            suitable_houses = [
                house for house in housing_market.houses 
                if house.price / house.square_feet <= self.annual_income / 12  # monthly salary
            ]
        elif self.segment == Segment.AVERAGE:
            # Look for houses with prices below average price in the market
            suitable_houses = [house for house in housing_market.houses if house.price <= average_price]
        else:
            suitable_houses = []
        
        # Check if there are suitable houses
        if not suitable_houses:
            print(f"Consumer {self.id} could not find a suitable house.")
            return
        
        # Check if consumer can afford any of the suitable houses
        for house in suitable_houses:
            # Assuming a 20% down payment requirement
            down_payment = house.price * 0.2
            if self.savings >= down_payment:
                # Successfully purchase house
                self.house = house
                self.savings -= down_payment  # Deduct down payment from savings
                print(f"Consumer {self.id} bought house {house.id} for ${house.price}.")
                return
        
        print(f"Consumer {self.id} cannot afford any suitable houses based on savings.")
