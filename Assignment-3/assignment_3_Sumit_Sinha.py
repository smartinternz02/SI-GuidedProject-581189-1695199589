# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Download the dataset: Dataset

# 2. Load the dataset into the tool
df = pd.read_csv('C:/Users/Sumit/AI ML SmartBridge/Assignment-3/penguins_size.csv')

# 3. Perform Below Visualizations

# Visualize the distribution of the target variable (Species)
sns.countplot(data=df, x='species')
plt.title('Distribution of Penguin Species')
plt.show()

# Pairplot for numeric variables
sns.pairplot(df, hue='species')
plt.title('Pairplot of Numeric Variables')
plt.show()

# 4. Perform descriptive statistics on the dataset
descriptive_stats = df.describe()
print(descriptive_stats)

# 5. Check for Missing values and deal with them
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# 6. Find the outliers and replace them

# Select only numeric columns for outlier detection
numeric_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Define a function to find and replace outliers using Z-scores
def replace_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = z_scores > threshold
    data[column][outliers] = np.nan

# Iterate through numeric columns and replace outliers using Z-scores
for col in numeric_cols:
    replace_outliers_zscore(df, col)

# Replace NaN values with the median of each column
df.fillna(df.median(), inplace=True)

# Now, df contains the dataset with outliers replaced using Z-scores
# 7. Check the correlation of independent variables with the target
correlation_matrix = df.corr()
target_correlation = correlation_matrix['body_mass_g'].sort_values(ascending=False)
print("Correlation with Target:")
print(target_correlation)

# 8. Check for Categorical columns and perform encoding
# The 'Sex' and 'Island' columns are categorical. You can encode them using one-hot encoding.
df_encoded = pd.get_dummies(df, columns=['sex', 'island'], drop_first=True)

# 9. Split the data into dependent and independent variables
X = df_encoded.drop('species', axis=1)
y = df_encoded['species']

# 10. Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 11. Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 12. Check the training and testing data shape
print("Training Data Shape:")
print(X_train.shape, y_train.shape)
print("Testing Data Shape:")
print(X_test.shape, y_test.shape)