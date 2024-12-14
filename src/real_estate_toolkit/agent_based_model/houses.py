from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QualityScore(Enum):
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1

@dataclass
class House:
    id: int
    price: float
    area: float
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True
    
    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.
        
        Divides the price by the area, rounded to two decimal places.
        """
        if self.area == 0:
            return 0.0  # Avoid division by zero
        return round(self.price / self.area, 2)
    
    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).
        
        Returns True if the house was built within the last 5 years.
        """
        return (current_year - self.year_built) < 5
    
    def get_quality_score(self) -> QualityScore:
        """
        Generate a quality score based on house attributes.

        If no quality score is provided, calculate a score based on size, bedrooms, and age.
        """
        if self.quality_score is None:  # Solo calcular si quality_score no está definido
            score = 3  # Default average score
            if self.area >= 2500:
              score += 1  # Larger houses tend to have better quality
            if self.bedrooms >= 3:
                score += 1  # More bedrooms usually indicate better quality
            if self.year_built >= 2010:
                score += 1  # Newer houses tend to have better quality

        # Determina el QualityScore basado en el puntaje calculado
            if score >= 5:
                self.quality_score = QualityScore.EXCELLENT
            elif score == 4:
                self.quality_score = QualityScore.GOOD
            elif score == 3:
                self.quality_score = QualityScore.AVERAGE
            elif score == 2:
                self.quality_score = QualityScore.FAIR
            else:
                self.quality_score = QualityScore.POOR

        return self.quality_score


    def sell_house(self) -> None:
        """
        Mark the house as sold by setting availability to False.
        """
        self.available = False

