# 1. IMPORT REQUIRED LIBRARIES
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. CREATE THE DATASET (MINI MOVIE/ANIME DATABASE)
data = {
    'Title': ['Oshi no Ko', 'Classroom of the Elite', 'Death Note', 'Naruto', 'True Detective'],
    'Description': [
        'The story of an idol\'s children uncovering the dark secrets of the show and entertainment business.',
        'The struggle of brilliant students surviving in an elite school through fierce mind games.',
        'A brilliant high school student uses a supernatural notebook to punish criminals, entering a battle of wits with the police.',
        'An action-packed adventure of a young ninja dreaming of becoming the village leader.',
        'Two detectives attempting to solve a mysterious and dark murder case over the years.'
    ]
}
df = pd.DataFrame(data)

# 3. TEXT VECTORIZATION (TF-IDF)
tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(df["Description"])

# 4. CALCULATE COSINE SIMILARITY
similarity_score = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 5. FIND RECOMMENDATIONS FOR A SPECIFIC SHOW
classroom_scores = list(enumerate(similarity_score[1]))

sorted_scores = sorted(classroom_scores, key=lambda x: x[1], reverse=True)

print("--- RECOMMENDATION ENGINE ---")
print(f"Suggestions for '{df['Title'][1]}' lovers:")

# 6. EXTRACT AND DISPLAY THE RESULT
most_similar_index = sorted_scores[1][0] 

print(f"> {df['Title'][most_similar_index]}")