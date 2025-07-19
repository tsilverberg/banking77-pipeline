from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer