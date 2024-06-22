from flask import Flask, request, render_template
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import docx2txt
from PyPDF2 import PdfReader
import re

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask application
app = Flask(__name__)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    tokens = word_tokenize(text.lower())  # Tokenization
    tokens = [word for word in tokens if word.isalnum()]  # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return tokens

def read_text_from_file(file):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension == '.txt':
        return file.read().decode('utf-8')
    elif file_extension == '.docx':
        return docx2txt.process(file)
    elif file_extension == '.pdf':
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    text = read_text_from_file(file)
    
    if text is None:
        return render_template('index.html', error='Invalid file type. Please upload a txt, docx, or pdf file.')

    tokens = preprocess_text(text)
    
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

    topics = lda_model.print_topics(num_words=5)
    
    explanation = generate_explanation(text, tokens, topics)

    return render_template('index.html', explanation=explanation)

def generate_explanation(text, tokens, topics):
    explanation = f"""
    <h3>Step-by-Step Explanation</h3>
    <ol>
        <li><strong>Text Preprocessing</strong><br>
            The given text is preprocessed before being fed into the LDA model. This preprocessing typically involves:
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
    return explanation

if __name__ == '__main__':
    app.run(debug=True)
