import pandas as pd
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('dataset.csv')

# Check the class distribution
print(df['label'].value_counts())

# Separate the classes
df_joy = df[df['label'] == 'joy']
df_happy = df[df['label'] == 'happy']
df_neutral = df[df['label'] == 'neutral']
df_angry = df[df['label'] == 'angry']
df_optimistic = df[df['label'] == 'optimistic']

# Perform oversampling on 'neutral' and 'optimistic' classes
df_neutral_upsampled = resample(df_neutral, 
                                replace=True,     # Sample with replacement
                                n_samples=df_joy.shape[0], # Match the number of 'joy' samples
                                random_state=42)  # Set random seed for reproducibility

df_optimistic_upsampled = resample(df_optimistic, 
                                   replace=True,     
                                   n_samples=df_joy.shape[0], 
                                   random_state=42)

# Combine the upsampled data with the rest
df_balanced = pd.concat([df_joy, df_happy, df_angry, df_neutral_upsampled, df_optimistic_upsampled])

# Shuffle the dataset to mix up the entries
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new class distribution
print(df_balanced['label'].value_counts())

# Save the balanced dataset
df_balanced.to_csv('balanced_dataset.csv', index=False)
