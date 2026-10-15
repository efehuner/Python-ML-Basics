# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 2. CREATE THE USER-ITEM RATING DATASET
data = {
    'User': ['Efe', 'Efe', 'Efe', 'Ali', 'Ali', 'Mehmet', 'Mehmet', 'Mehmet'],
    'Series': ['Oshi no Ko', 'Death Note', 'True Detective', 'Death Note', 'True Detective', 'Oshi no Ko', 'Classroom of the Elite', 'Death Note'],
    'Score': [9, 10, 8, 9, 9, 8, 10, 7]
}
df = pd.DataFrame(data)

# 3. CREATE A PIVOT TABLE (USER-ITEM MATRIX)
# Rows: Users, Columns: Series, Values: Scores
matrix = df.pivot_table(index="User", columns="Series", values="Score")

# Fill NaN (Not a Number) values with 0 (meaning the user hasn't watched/rated it)
matrix = matrix.fillna(0)
print("--- USER-SERIES SCORE MATRIX ---\n")
print(matrix, "\n")

# 4. CALCULATE PEARSON CORRELATION (COLLABORATIVE FILTERING)
# Calculates how similar the rating behaviors are between different series
similarity_series = matrix.corr()

# 5. VISUALIZE THE CORRELATION MATRIX WITH A HEATMAP
plt.figure(figsize=(12, 8))
plt.title("Similarity of Series (Correlation Matrix)")
# RdYlGn colormap shows high correlation in Green, negative in Red
sns.heatmap(similarity_series, cmap="RdYlGn", annot=True, vmin=-1, vmax=1, annot_kws={"fontsize": 11, "fontweight": "bold"})
plt.show()

# 6. GET RECOMMENDATIONS FOR A SPECIFIC SERIES
# Sort the correlation scores for 'Death Note' in descending order
death_note_suggestion = similarity_series["Death Note"].sort_values(ascending=False)

print("--- RECOMMENDATION ENGINE ---")
print("Suggestions for 'Death Note' Lovers:")
# Start from index 1 to skip 'Death Note' itself (which has a correlation of 1.0)
print(death_note_suggestion[1:4])