from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import uuid
import speech_recognition as sr
import spacy
import json
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import numpy as np
import os
from dotenv import load_dotenv
import PyPDF2
from collections import deque
import time
from PIL import Image as PIL_Image
import pytesseract
import google.generativeai as genai
from flask_cors import CORS  # Import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads' # Create this folder in the same directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# 1. NLP Module
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # If not installed, download the model
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def process_query(text, nlp):
    doc = nlp(text)
    keywords = [token.text for token in doc if not token.is_stop and token.is_alpha] # Get tokens that are words and not stop words
    return keywords

# 2. Speech Recognition Module
def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio) # Can change to other Speech recognition API providers like Whisper
        print(f"Voice input recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# 3. Knowledge Base Loading Module
def load_knowledge_base(filepath):
     with open(filepath, 'r') as file:
            return json.load(file)

def search_knowledge_base(keywords, knowledge_base):
    """
    Searches the knowledge base for the most relevant content based on keywords.

    Args:
      keywords (list): A list of keywords extracted from the user's question.
      knowledge_base (dict): The loaded knowledge base from a JSON file.

    Returns:
        tuple: A tuple containing:
            - str or None: The most relevant answer from the knowledge base, or None if no match is found.
            - str or None: The category of the topic
    """

    best_match = None
    best_score = 0
    best_category = None

    for category, entries in knowledge_base.items():
      if category != "PDF": # Do not search in PDF section, it will be done later
        for entry in entries:
            if 'keywords' in entry and 'answer' in entry:
                entry_keywords = entry['keywords']
                # Calculate the Jaccard similarity (ratio of intersection to union)
                score = len(set(keywords).intersection(entry_keywords)) / len(set(keywords).union(entry_keywords)) if len(set(keywords).union(entry_keywords)) > 0 else 0

                if score > best_score:
                    best_score = score
                    best_match = entry['answer']
                    best_category = category

    if best_match:
        return best_match, best_category
    else:
        return None, None

# 4. Gemini Language Model Module
class Gemini_Assistant:
    def __init__(self, model="gemini-pro", api_key=None):
        """
        Initializes the Gemini_Assistant with an API key and a model.

        Args:
            model (str): The model to use for generating the answer, default is gemini-pro.
            api_key (str): The Google API key, if not given will read from environment variables.
        """

        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("API Key not found. Please set the GOOGLE_API_KEY environment variable.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.conversation = deque(maxlen=5)  # Keep track of the last few exchanges

    def generate_answer(self, question, context=None):
        """
         Generate answer from a question using Gemini API.

        Args:
            question (str): A string of question
            context (str): Additional content to pass as context to Gemini

         Returns:
            str: answer text
        """
        prompt = question
        if context:
          prompt = f"Given the following context: {context}. Please provide a detailed answer to the user query: {question}. Format your response in a clear and understandable way. Remove any bullet points, `*` or other separating characters. If the answer is best presented as a table then output a table or if the answer is best represented with an image then give a markdown link for the image."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error while using Gemini API {e}")
            return None


    def clear_conversation(self):
        self.conversation = deque(maxlen=5) # clear old conversations.

# 5. PDF Generation Module
def create_pdf(question, answer, file_path):
    """
    Generates a PDF file containing the question and the answer.
    
    Args:
        question (str): A string of the question asked by the user.
        answer (str): A string of the generated answer.
        file_path (str): A string of file path to save the PDF.
    """
    results_folder = "results" # Set the path here
    os.makedirs(results_folder, exist_ok = True) # Create results folder if it does not exist.
    pdf_doc = SimpleDocTemplate(os.path.join(results_folder, file_path), pagesize=letter) # add the results folder
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Question:</b>", styles['h3']))
    elements.append(Paragraph(question, styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Answer:</b>", styles['h3']))
    elements.append(Paragraph(answer, styles['Normal']))
    
    pdf_doc.build(elements)

# 6. OCR Module
def extract_text_from_image(image_path):
    """
    Extracts text from an image using pytesseract.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text or None if error.
    """
    try:
        img = PIL_Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print(f"Text extracted from image: {text}")
        return text.strip()
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

def search_pdf_content(pdf_text, keywords):
    """
        Searches for relevant answer to a question inside a pdf document.

        Args:
            pdf_text (str): Text of the PDF Document
            keywords (list): List of keywords for your search

        Returns:
            (str) : Returns the answer of the question or None
    """
    best_match = None
    best_score = 0

    # Split the text into sentences or paragraphs to search
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', pdf_text)  # Split by end of sentences
    for sentence in sentences:
       score = len(set(keywords).intersection(sentence.split())) / len(set(keywords).union(sentence.split())) if len(set(keywords).union(sentence.split())) > 0 else 0
       if score > best_score:
            best_score = score
            best_match = sentence
    if best_match:
        return best_match
    else:
       return None

# Main Function for Processing User Queries
def process_user_query(nlp, assistant, knowledge_base, query, is_voice=False, image_path=None, file_path=None):
    """
    Processes user queries, both text, voice and image based.

    Args:
        nlp (spaCy Language model): spaCy language model
        assistant (Gemini_Assistant): Gemini Assistant
        knowledge_base (dict): The loaded knowledge base
        query (str): A string of the question from the user or extracted from the image
        is_voice (bool): A boolean of whether the query was from voice
        image_path (str): A string of image path if the input is an image
        file_path (str): A string of file path if the input is a pdf or docx file
    Returns:
        (str, str): String of answer and pdf file path
    """
    pdf_text_from_file = None
    if image_path:
        image_text = extract_text_from_image(image_path)
        if image_text:
            query = image_text
        else:
            return "I'm sorry, I cannot process the image at the moment.", None
    elif file_path:
        # You would put your logic for processing a pdf/docx here (extracting text, etc).
        try:
            if file_path.lower().endswith(".pdf"):
                pdf_text_from_file = extract_text_from_pdf(file_path) # Save pdf text from the file to use later.
            # You may add for docx processing as well
            else:
                return "I'm sorry, I cannot process the file at the moment", None
        except Exception as e:
             return f"I'm sorry, there was an error processing the file {e}", None


    if not query:
        return "I did not receive a valid query", None
    
    keywords = process_query(query, nlp)
    if not keywords:
        return "I did not understand the query, can you please rephrase it", None
    
    answer, category = search_knowledge_base(keywords, knowledge_base) # Search in JSON knowledge base first.
    if not answer and pdf_text_from_file:  # If no answer from JSON, try PDF content
        answer = search_pdf_content(pdf_text_from_file, keywords)
    # use Gemini for generating a better formatted response from the extracted results
    if answer:
      answer = assistant.generate_answer(query, context = answer)
    elif not answer:
        print("No answer in knowledge base, trying Gemini API.")
        answer = assistant.generate_answer(query)
    if not answer:
        return "I'm sorry, I cannot answer the question", None

    pdf_file_name = f"answer_{int(time.time())}.pdf" # Generates unique pdf file name using time.
    create_pdf(query, answer, pdf_file_name) # removed `os.path.join(UPLOAD_FOLDER,pdf_file_name)`
    return answer, pdf_file_name


def extract_text_from_pdf(file_path):
    """
     Extract text from a PDF file
     Args:
         file_path (str) : Path to PDF file
     Returns:
        (str): Extracted Text from PDF file
    """
    try:
      pdfFileObj = open(file_path, 'rb')
      pdfReader = PyPDF2.PdfReader(pdfFileObj)
      num_pages = len(pdfReader.pages)
      text = ""
      for page in range(num_pages):
          pageObj = pdfReader.pages[page]
          text += pageObj.extract_text()
      return text.strip()
    except Exception as e:
        print(f"Error reading PDF : {e}")
        return None

nlp = load_nlp_model()
knowledge_base = load_knowledge_base('knowledge_base.json')
assistant = Gemini_Assistant()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_request():
    if 'file' in request.files:
        files = request.files.getlist('file') # Get all files uploaded
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                if file_path.lower().endswith(".pdf"):
                    pdf_text = extract_text_from_pdf(file_path)
                    if pdf_text:
                        # Add the extracted text to knowledge base here
                        if not any(entry.get('pdf_filename') == filename for category in knowledge_base.values() for entry in category): # Avoid repeated PDF uploads.
                            knowledge_base["PDF"] = knowledge_base.get("PDF",[]) + [{"pdf_filename": filename, "content": pdf_text}] # Add to knowledge base with PDF
                            # Save knowledge_base with the new entry
                            with open('knowledge_base.json', 'w') as f:
                                json.dump(knowledge_base, f, indent=4)

                if os.path.exists(file_path):
                   os.remove(file_path) # Delete the file after processing
            else:
                return jsonify({'answer' : "Error reading the file"}), 400

        answer, pdf_file = process_user_query(nlp, assistant, knowledge_base, query=None)

    elif request.is_json:
        data = request.get_json()
        query = data.get("query")
        input_type = data.get("input_type", "text")
        if input_type == 'voice':
            audio_file_name = f"audio_file_{int(time.time())}.wav"
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file_name)
             # Create a temporary file for the voice input
            with open(audio_file_path, "w") as temp_audio:
              pass # touch the file and create it.
            text = recognize_speech(audio_file_path)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            if text:
              answer, pdf_file = process_user_query(nlp, assistant, knowledge_base, query=text)
            else:
                 return jsonify({'answer' : "Could not understand speech"}), 400
        elif query:
            answer, pdf_file = process_user_query(nlp, assistant, knowledge_base, query = query)
        else:
            return jsonify({'answer' : "Could not understand the request"}), 400
    else:
       return jsonify({'answer' : "Invalid request"}), 400

    return jsonify({'answer': answer, "pdf_file": pdf_file })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
   file_path = os.path.join("results", filename) # add results here as the path where the PDF was saved.
   if os.path.exists(file_path):
       return send_file(file_path, as_attachment = True)
   return jsonify({'error' : "File not found"}), 404

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'docx'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)