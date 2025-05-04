##########################################################################################
# Imports
##########################################################################################



##########################################################################################
# Data Transformation Methods
##########################################################################################
def split_dataset(dataset, train_fraction=0.8):
    """
    Creates a deep copy of the dataset and calculates the number of rows to be used for training,
    based on a specified fraction (default is 80%).

    Parameters:
      dataset (pd.DataFrame): The input dataset.
      train_fraction (float): Fraction of data to use for training (between 0 and 1).

    Returns:
      pd.DataFrame: A deep copy of the dataset.
      int: The number of rows to be used for training.
    """

    # Create a deep copy of the dataset to avoid modifying the original data
    X = dataset.copy(deep=True)

    # Get the total number of rows in the dataset
    row = len(X)

    # Compute the number of rows that will be used for training
    train_len = int(row * train_fraction)

    return X, train_len
