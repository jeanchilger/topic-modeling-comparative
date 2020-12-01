from preprocessing.preprocess import Preprocessor

preprocessor = Preprocessor(
        "data/news.csv",
        columns=["title", "description", "text"])

print(preprocessor.raw_corpus[0][0])

print()
print("==========================================")
print()

print(preprocessor.tokens[0])
print(preprocessor.ngram_corpus[0])