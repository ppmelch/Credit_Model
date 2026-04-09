import pandas as pd


class DataPreparation:
    """
    Class responsible for preparing the dataset for modeling.

    This includes:
    - Separating features (X) and target (y)
    - Encoding categorical variables
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the data preparation object.

        Parameters
        ----------
        data : pd.DataFrame
            Raw input dataset.
        """
        self.data = data.copy()

    def prepare_data(self):
        """
        Prepare the dataset for modeling.

        This method:
        - Extracts the target variable ('loan_status')
        - Removes non-feature columns (e.g., 'interest_rate')
        - Encodes categorical variables using one-hot encoding

        Returns
        -------
        X : pd.DataFrame
            Processed feature matrix
        y : pd.Series
            Target variable
        """
        # Target
        y = self.data['loan_status']

        # Features
        X = self.data.drop(columns=['loan_status', 'interest_rate'])

        # Encoding
        X = pd.get_dummies(X, drop_first=True)

        return X, y
