from flask import Flask, request, render_template
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import docx2txt
from PyPDF2 import PdfReader

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask application
app = Flask(__name__)

# Homepage route - renders index.html for file upload
@app.route('/')
def index():
    return render_template('index.html')

# Route for file upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    # Retrieve uploaded file from form
    file = request.files['file']
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Check file type and process accordingly
    if file_extension == '.txt':
        text = file.read().decode('utf-8')
    elif file_extension == '.docx':
        text = docx2txt.process(file)
    elif file_extension == '.pdf':
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        return render_template('index.html', error='Invalid file type. Please upload a txt, docx, or pdf file.')

    # Tokenization and stop word removal
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary([filtered_tokens])
    corpus = [dictionary.doc2bow(filtered_tokens)]

    # Build LDA model
    lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary)

    # Get topics and their probabilities
    topics = lda_model.show_topics(formatted=False)

    # Extract file name and type
    filename = file.filename
    filetype = file_extension[1:].upper()  # Remove dot from extension and convert to uppercase

    # Render results in HTML template with file name, type, and topics
    return render_template('index.html', topics=topics, filename=filename, filetype=filetype)

if __name__ == '__main__':
    app.run(debug=True)
