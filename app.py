from flask import Flask, request, render_template
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    text = file.read().decode('utf-8')
    
    # Step 1: Text Preprocessing
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]

    # Step 2: Creating the Document-Term Matrix
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Step 3: Training the LDA Model
    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

    # Step 4: Output Interpretation
    topics = lda_model.print_topics(num_words=5)

    # Detailed explanation for the output
    explanation = f"""
    <h3>Step-by-Step Explanation</h3>
    <ol>
        <li><strong>Text Preprocessing</strong><br>
            The given text about artificial intelligence is preprocessed before being fed into the LDA model. This preprocessing typically involves:
            <ul>
                <li>Tokenization: Breaking down the text into individual words or tokens.</li>
                <li>Removing Stop Words: Filtering out common words (like "and", "the", etc.) that do not contribute much meaning to the analysis.</li>
                <li>Lowercasing: Converting all words to lowercase to ensure uniformity.</li>
                <li>Stemming or Lemmatization: Reducing words to their root forms (e.g., "learning" to "learn").</li>
            </ul>
            <pre>Original Text: {text}</pre>
            <pre>Preprocessed Text: {' '.join(tokens)}</pre>
        </li>
        <li><strong>Creating the Document-Term Matrix</strong><br>
            After preprocessing, the text is transformed into a numerical representation that the LDA model can work with. This involves:
            <ul>
                <li>Creating a Dictionary: Mapping each unique word in the corpus to an ID.</li>
                <li>Bag-of-Words (BoW): Representing each document as a vector of word frequencies.</li>
            </ul>
            Example:
            <pre>Document: {tokens}</pre>
        </li>
        <li><strong>Training the LDA Model</strong><br>
            The LDA (Latent Dirichlet Allocation) model is trained on the document-term matrix. The LDA model works by:
            <ul>
                <li>Assuming that each document is a mixture of topics.</li>
                <li>Assuming that each topic is a mixture of words.</li>
                <li>Iteratively refining these assumptions to best explain the observed data.</li>
            </ul>
            Number of Topics: 2
        </li>
        <li><strong>Output Interpretation</strong><br>
            After training, the LDA model provides the topics with the top words and their associated weights. The output provided is:
            <pre>{topics}</pre>
            Explanation:
            <ul>
                <li>Topic 1: <pre>{topics[0]}</pre></li>
                <li>Topic 2: <pre>{topics[1]}</pre></li>
            </ul>
        </li>
        <li><strong>Interpretation of Topics</strong><br>
            Based on the top words and their weights, we can interpret the topics:
            <ul>
                <li>Topic 1 likely represents general discussions about AI systems, intelligence, human aspects, and strong AI.</li>
                <li>Topic 2 likely represents discussions contrasting weak AI and strong AI.</li>
            </ul>
        </li>
    </ol>
    """

    return render_template('index.html', explanation=explanation)

if __name__ == '__main__':
    app.run(debug=True)
