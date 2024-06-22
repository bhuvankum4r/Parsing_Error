# nlp_model.py
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import gensim

def preprocess_text(text):
    # Lowercase and remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text.lower())
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return lemmatized_tokens

def run_lda_model(tokens_list):
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    
    # Train LDA model
    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)
    
    # Extract topics
    topics = lda_model.print_topics(num_words=5)
    return topics

def process_text(text):
    processed_tokens = preprocess_text(text)
    topics = run_lda_model([processed_tokens])
    summary = "\n".join([f"Topic {i+1}: {topic}" for i, topic in enumerate(topics)])
    return summary
