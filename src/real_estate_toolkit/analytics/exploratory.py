from typing import List, Dict, Any, Optional
import polars as pl
import plotly.express as px

path = "C:/SARA/project"

class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.real_state_data = pl.read_csv(f"{path}/files/test.csv")
        self.real_state_clean_data = None

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning:
        
        Tasks to implement:
        1. Identify and handle missing values
            - Decide strategy for each column (drop, fill with mean/median, no change)
        2. Convert columns to appropriate data types if needed.
            - Ensure numeric columns are numeric
            - Ensure categorical columns are categorized
        
        Returns:
            Cleaned and preprocessed dataset assigned to self.real_state_clean_data
        """
        self.real_state_clean_data = self.real_state_data.clone()

        for col in self.real_state_clean_data.columns:
            if self.real_state_clean_data[col].null_count() > 0:
                if self.real_state_clean_data[col].dtype == pl.Float64 or self.real_state_clean_data[col].dtype == pl.Int64:
                    median_value = self.real_state_clean_data[col].median()
                    self.real_state_clean_data = self.real_state_clean_data.with_column(
                        self.real_state_clean_data[col].fill_null(median_value)
                    )


    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        
        Tasks to implement:
        1. Compute basic price statistics and generate another data frame called price_statistics:
            - Mean
            - Median
            - Standard deviation
            - Minimum and maximum prices
        2. Create an interactive histogram of sale prices using Plotly.
        
        Returns:
            - Statistical insights dataframe
            - Save Plotly figures for price distribution in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Clean data is not available. Please run clean_data() first.")

        required_columns = ["SalePrice", "GrLivArea"]
        for col in required_columns:
            if col not in self.real_state_clean_data.columns:
                raise ValueError(f"Falta la columna {col} en los datos")

        price_col = "SalePrice"
        price_statistics = pl.DataFrame({
            "Mean": [self.real_state_clean_data[price_col].mean()],
            "Median": [self.real_state_clean_data[price_col].median()],
            "StdDev": [self.real_state_clean_data[price_col].std()],
            "Min": [self.real_state_clean_data[price_col].min()],
            "Max": [self.real_state_clean_data[price_col].max()]
        })

        # Create histogram
        fig = px.histogram(
            self.real_state_clean_data.to_pandas(),
            x=price_col,
            title="Sale Price Distribution",
            labels={"x": "Sale Price"},
            nbins=50
        )
        fig.update_layout(bargap=0.2)

        # Save figure
        output_path = f"{path}/real_estate_toolkit/src/real_estate_toolkit/analytics/outputs/sale_price_distribution.html"
        fig.write_html(output_path)        

        return price_statistics

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        
        Tasks to implement:
        1. Group data by neighborhood
        2. Calculate price statistics for each neighborhood
        3. Create Plotly boxplot with:
            - Median prices
            - Price spread
            - Outliers
        
        Returns:
            - Return neighborhood statistics dataframe
            - Save Plotly figures for neighborhood price comparison in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            self.clean_data()
            raise ValueError("Clean data is not available. Please run clean_data() first.")

        neighborhood_col = "Neighborhood"
        price_col = "SalePrice"

        # Group by neighborhood and calculate statistics
        neighborhood_stats = (
            self.real_state_clean_data
            .groupby(neighborhood_col)
            .agg([
                pl.col(price_col).mean().alias("MeanPrice"),
                pl.col(price_col).median().alias("MedianPrice"),
                pl.col(price_col).std().alias("StdDevPrice"),
                pl.col(price_col).min().alias("MinPrice"),
                pl.col(price_col).max().alias("MaxPrice")
            ])
        )

        # Create boxplot
        fig = px.box(
            self.real_state_clean_data.to_pandas(),
            x=neighborhood_col,
            y=price_col,
            title="Neighborhood Price Comparison",
            labels={"x": "Neighborhood", "y": "Sale Price"},
        )

        # Save figure
        output_path = f"{path}/real_estate_toolkit/src/real_estate_toolkit/analytics/outputs/neighborhood_price_comparison.html"
        fig.write_html(output_path)
        
        
        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for variables input.
        
        Tasks to implement:
        1. Pass a list of numerical variables
        2. Compute correlation matrix and plot it
        
        Args:
            variables (Lis[str]): List of variables to correlate
        
        Returns:
            Save Plotly figures for correlation heatmap in src/real_estate_toolkit/analytics/outputs/ folder.
        """
        if self.real_state_clean_data is None:
            raise ValueError("Clean data is not available. Please run clean_data() first.")

        # Compute correlation matrix
        corr_matrix = self.real_state_clean_data.select(variables).to_pandas().corr()

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Heatmap",
            labels=dict(color="Correlation"),
        )

        # Save figure
        output_path = f"{path}/real_estate_toolkit/src/real_estate_toolkit/analytics/outputs/correlation_heatmap.html"
        fig.write_html(output_path)  
            

    def create_scatter_plots(self) -> Dict[str, Any]:
        """
        Create scatter plots exploring relationships between key features.
        
        Scatter plots to create:
        1. House price vs. Total square footage
        2. Sale price vs. Year built
        3. Overall quality vs. Sale price
        
        Tasks to implement:
        - Use Plotly Express for creating scatter plots
        - Add trend lines
        - Include hover information
        - Color-code points based on a categorical variable
        - Save them in in src/real_estate_toolkit/analytics/outputs/ folder.
        
        Returns:
            Dictionary of Plotly Figure objects for different scatter plots. 
        """
        if self.real_state_clean_data is None:
            raise ValueError("Clean data is not available. Please run clean_data() first.")

        plots = {}

        scatter_configs = [
            {"x": "GrLivArea", "y": "SalePrice", "color": "Neighborhood", "title": "Price vs. Living Area"},
            {"x": "YearBuilt", "y": "SalePrice", "color": "Neighborhood", "title": "Price vs. Year Built"},
            {"x": "OverallQual", "y": "SalePrice", "color": "Neighborhood", "title": "Price vs. Overall Quality"},
        ]

        for config in scatter_configs:
            fig = px.scatter(
                self.real_state_clean_data.to_pandas(),
                x=config["x"],
                y=config["y"],
                color=config["color"],
                trendline="ols",
                title=config["title"],
                labels={"x": config["x"], "y": config["y"]},
            )

            # Save figure
            output_path = f"{path}/real_estate_toolkit/src/real_estate_toolkit/analytics/outputs/scatter_{config['x']}_vs_{config['y']}.html"
            fig.write_html(output_path)

            plots[f"{config['x']}_vs_{config['y']}"] = fig

        return plots
