import pandas as pd
import numpy as np

df = pd.read_csv('reviews.csv')
# print(df)

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


#Feature extraction

x = df['Reviews']
# print(x)

vector = CountVectorizer()
vector.fit(x)
# print(vector)

vector_X = vector.transform(x)
# print(vector_X)

# print(vector.vocabulary_) # the hash that the CountVectorizer gave to the Reviews (X)


# TFIDF extraction

tfidf = TfidfTransformer()
tfidf.fit(vector_X)
# print(tfidf)

review = tfidf.transform(vector_X) #will give like a precentage of frequancy of the word
# print(review)

y = df['Rating'].tolist()
# print(y)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(review,y)

txt = ["The product is not stisfactory"]

txt_ex = vector.transform(txt)
txt_tf = tfidf.transform(txt_ex)

# print(model.predict(txt_tf))

def rate(*comment):
    f_ex = vector.transform(comment)
    tf = tfidf.transform(f_ex)
    prediction = model.predict(tf)
    for review,ret in zip(comment,prediction):
        print(review,':\n','Rating: ', ret)


if __name__ == '__main__':
    rate(input("Review : "))