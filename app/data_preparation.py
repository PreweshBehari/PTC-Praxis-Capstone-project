##########################################################################################
# Imports
##########################################################################################
import streamlit as st


##########################################################################################
# Data Preparation Methods
##########################################################################################
def drop_columns_with_excessive_missing_data(dataset, threshold=0.3):
    """
    Identifies and drops columns from the dataset that have a fraction of missing values
    exceeding the specified threshold (default is 30%).

    Parameters:
      dataset (pd.DataFrame): The input dataset to process.
      threshold (float): The cutoff proportion for missing data. Columns with more missing
                       data than this value will be dropped.

    Returns:
      pd.DataFrame: The dataset with specified columns removed.
    """
    st.subheader("Data Cleaning")

    # 1. Calculate the fraction of missing values for each column,
    # and sort the result in descending order (highest missing fraction first).
    missing_fractions = dataset.isnull().mean().sort_values(ascending=False)

    # 2. View the top 10 columns with the most missing values
    st.write("Top 10 columns with missing values:")
    st.write(missing_fractions.head(10))

    # 3. Identify columns where more than 'threshold' fraction of data is missing
    drop_list = sorted(list(missing_fractions[missing_fractions > threshold].index))

    # 4. Drop those columns from the dataset in-place
    dataset.drop(labels=drop_list, axis=1, inplace=True)

    # 5. Return the new shape of the dataset after dropping columns
    st.write(f"Dropped {len(drop_list)} columns with more than {threshold*100}% missing data")
    st.write(f"New dataset shape: {dataset.shape}")

    return dataset
