import pandas as pd
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handle data preprocessing tasks."""

    def __init__(self):
        self.scalers = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df (pd.DataFrame): The dataframe to process.

        Returns:
            pd.DataFrame: The dataframe with missing values handled.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Linear interpolation for numeric columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        # Fill remaining NaN values with 0 as fallback
        df = df.fillna(0)
        return df

    def detect_and_remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Tüm sayısal sütunlarda aykırı değerleri tespit eder ve temizler.

        Args:
            df (pd.DataFrame): Veri çerçevesi.
            method (str): Aykırı değer temizleme yöntemi ('iqr').

        Returns:
            pd.DataFrame: Aykırı değerleri temizlenmiş veri çerçevesi.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns  # Sayısal sütunları seç
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Aykırı değerleri filtrele
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        return df

    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Scale features using the specified method.

        Args:
            df (pd.DataFrame): The dataframe to process.
            columns (List[str]): The columns to scale.
            method (str): The method to use for scaling ('standard').

        Returns:
            pd.DataFrame: The dataframe with scaled features.
        """
        if method == 'standard':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers['standard'] = scaler
        return df
