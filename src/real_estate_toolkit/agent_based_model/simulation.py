from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, shuffle
from typing import List, Dict, Any
from .houses import House
from .house_market import HousingMarket
from .consumers import Segment, Consumer

class CleaningMarketMechanism(Enum):
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()

@dataclass
class AnnualIncomeStatistics:
    minimum: float
    average: float
    standard_deviation: float
    maximum: float

@dataclass
class ChildrenRange:
    minimum: int = 0
    maximum: int = 5

@dataclass
class Simulation:
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def __post_init__(self):
        """
        Automatically initialize the housing market and consumers after instantiation.
        
        This method is useful for ensuring all dependent attributes are initialized
        without requiring explicit calls by the user after creating an instance.
        """
        self.housing_market = self.create_housing_market()
        self.consumers = self.create_consumers()

    def create_housing_market(self) -> HousingMarket:
        """
        Initialize market with houses.
        """
        houses = [House(**house_data) for house_data in self.housing_market_data]
        if not houses:
            raise ValueError("Markets can't be created without houses.")
        return HousingMarket(houses)

    def create_consumers(self) -> List[Consumer]:
        """
        Generate consumer population.
        """
        consumers = []
        for _ in range(self.consumers_number):
            # Generate random annual income within the specified range
            while True:
                income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
                if self.annual_income.minimum <= income <= self.annual_income.maximum:
                    break

            # Generate random number of children within the range
            children = randint(self.children_range.minimum, self.children_range.maximum)

            # Randomly assign a segment
            segment = Segment(randint(0, len(Segment) - 1))

            # Create a consumer
            consumer = Consumer(
                annual_income=income,
                children=children,
                segment=segment,
                saving_rate=self.saving_rate
            )
            consumers.append(consumer)
        return consumers

    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers.
        """
        for consumer in self.consumers:
            consumer.savings = consumer.annual_income * self.saving_rate

    def clean_the_market(self) -> None:
        """
        Execute market transactions.
        """
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)

        for consumer in self.consumers:
            self.housing_market.buy_a_house(consumer)

    def compute_owners_population_rate(self) -> float:
        """
        Compute the owners population rate after the market is clean.
        """
        owners = sum(1 for consumer in self.consumers if consumer.owns_house)
        return owners / self.consumers_number

    def compute_houses_availability_rate(self) -> float:
        """
        Compute the houses availability rate after the market is clean.
        """
        available_houses = len([house for house in self.housing_market.houses if not house.is_sold])
        total_houses = len(self.housing_market.houses)
        return available_houses / total_houses
