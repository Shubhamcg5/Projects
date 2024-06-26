# -*- coding: utf-8 -*-
"""recommended_system_books.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Wh6YFWRoQA7hV5nzs_xksrLN4nU6bRxJ
"""

import pandas as pd
df=pd.read_csv("book (1).csv")
df

df.info()

df.duplicated().sum()

df.head()

# Create a ratings matrix
ratings_matrix = df.pivot_table(index='User.ID', columns='Book.Title', values='Book.Rating', fill_value=0)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(ratings_matrix, ratings_matrix)

# Function to get top N similar books based on cosine similarity
def get_top_similar_books(book_title, similarity_matrix, top_n=2):
    book_index = df[df['Book.Title'] == book_title].index[0]
    similar_books = list(enumerate(similarity_matrix[book_index]))
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
    similar_books = similar_books[1:top_n + 1]  # Exclude the book itself

    top_books = [df['Book.Title'][i[0]] for i in similar_books]
    top_scores = [i[1] for i in similar_books]

    return top_books, top_scores

book_to_recommend = 'Classical Mythology'
top_books, top_scores = get_top_similar_books(book_to_recommend, cosine_sim, top_n=2)

print('book to recommend:',book_to_recommend)
print(top_books,top_scores)



