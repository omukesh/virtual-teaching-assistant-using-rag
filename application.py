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
import logging # Import logging module
from sentence_transformers import SentenceTransformer
import faiss

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = 'uploads' # Create this folder in the same directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. NLP Module
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logging.info("spaCy model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading spaCy model: {e}")
        from spacy.cli import download
        download("en_core_web_sm")  # Try to download if it fails
        nlp = spacy.load("en_core_web_sm")
    return nlp

def process_query(text, nlp):
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha] # Get tokens that are words and not stop words
    return keywords

# 2. Speech Recognition Module
def recognize_speech(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio) # Can change to other Speech recognition API providers like Whisper
        logging.info(f"Voice input recognized: {text}")
        return text
    except sr.UnknownValueError:
        logging.warning("Could not understand audio.")
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# 3. Knowledge Base Loading Module
def load_knowledge_base(filepath):  #Corrected load Knowledge Base
    """Loads the knowledge base from a JSON file."""
    try:
        with open(filepath, 'r') as file:
            knowledge_base = json.load(file)
            logging.info("Knowledge base loaded successfully.")
            return knowledge_base
    except FileNotFoundError:
            logging.error(f"Knowledge base file not found: {filepath}")
            return {}
    except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in knowledge base: {e}")
            return {}


# 4. Gemini Language Model Module
class Gemini_Assistant:
    def __init__(self, model="gemini-pro", api_key=None):
        """
        Initializes the Gemini_Assistant with an API key and a model.
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
        """
        prompt = question
        if context:
          prompt = f"Act as a helpful tutor. Given the following context: {context}. Please provide a detailed answer to the user query: {question}. Format your response in a clear and understandable way. Remove any bullet points, `*` or other separating characters. If the answer is best presented as a table then output a table or if the answer is best represented with an image then give a markdown link for the image."
        try:
            response = self.model.generate_content(prompt)
            self.conversation.append({"role": "user", "parts": [question]})
            self.conversation.append({"role": "model", "parts": [response.text]})
            return response.text
        except Exception as e:
            logging.error(f"Error while using Gemini API {e}")
            return None

    def clear_conversation(self):
        self.conversation.clear() # clear old conversations.
        logging.info("Conversation history cleared.")


# 5. PDF Generation Module
def create_pdf(question, answer, file_path):
    """
    Generates a PDF file containing the question and the answer.
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

    try:
        pdf_doc.build(elements)
        logging.info(f"PDF created successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error creating PDF: {e}")

# 6. OCR Module
def extract_text_from_image(image_path):
    """
    Extracts text from an image using pytesseract.
    """
    try:
        img = PIL_Image.open(image_path)
        text = pytesseract.image_to_string(img)
        logging.info(f"Text extracted from image: {text}")
        return text.strip()
    except Exception as e:
        logging.error(f"Error in OCR: {e}")
        return None

def search_pdf_content(pdf_text, keywords):
    """
        Searches for relevant answer to a question inside a pdf document.
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

# 7. FAISS Vector Search Setup
class KnowledgeBaseManager:
    def __init__(self, knowledge_base_path='knowledge_base.json', embedding_model_name='all-MiniLM-L6-v2', vector_dimension=384):  # vector_dimension depends on embedding model
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(self.embedding_model_name)
        self.knowledge_base = self.load_knowledge_base()
        self.index = faiss.IndexFlatIP(vector_dimension) #FAISS index
        self.ids = []  # Store IDs for document lookup
        self.metadata = [] # Metadata list
        self.build_index()


    def load_knowledge_base(self):  #Corrected load Knowledge Base
        """Loads the knowledge base from a JSON file."""
        try:
            with open(self.knowledge_base_path, 'r') as file:
                knowledge_base = json.load(file)
                logging.info("Knowledge base loaded successfully.")
                return knowledge_base
        except FileNotFoundError:
            logging.error(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON in knowledge base: {e}")
            return {}


    def build_index(self):
        """Builds a FAISS index from the knowledge base content."""
        documents = []
        ids = []
        id_counter = 0
        self.metadata = [] # Metadata list
        for category, entries in self.knowledge_base.items():
            if isinstance(entries, dict):  # Check if entries is a dictionary
              for topic, topic_entries in entries.items():
                for entry in topic_entries:
                    if 'content' in entry:
                        documents.append(entry['content'])  # Use the 'content' field
                        ids.append(id_counter)
                        self.metadata.append({"topic": topic, "category": category, "image_path": entry.get("image_path")}) # Store metadata
                        id_counter +=1

        if not documents:
            logging.warning("No content found in knowledge base to index.")
            return

        embeddings = self.model.encode(documents)
        self.index.add(embeddings) # Add the generated embeddings to the index.
        self.ids = ids # Keep the order of indexed documents.
        logging.info(f"FAISS index built with {len(documents)} documents.")

    def search(self, query, top_k=3):
        """Searches the FAISS index for the most relevant documents and retrieves metadata."""
        query_embedding = self.model.encode([query])
        D, I = self.index.search(query_embedding, top_k)  # D: distances, I: indices
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx != -1: # Check if valid
              metadata = self.metadata[idx]  # Get metadata using index
              results.append({
                  "content": documents[idx], # the "content" associated to the metadata
                  "metadata": metadata,
                  "distance": D[0][i]
              })
        return results

# Main Function for Processing User Queries
def process_user_query(nlp, assistant, kb_manager, query, is_voice=False, image_path=None, file_path=None):
    """Processes user queries, both text, voice and image based."""

    if not query:
        return "I did not receive a valid query", None

    keywords = process_query(query, nlp)
    if not keywords:
        return "I did not understand the query, can you please rephrase it", None

    # Perform semantic search using the KnowledgeBaseManager
    results = kb_manager.search(query)

    if results:
        best_result = results[0] # take top result
        context = best_result["content"]
        metadata = best_result["metadata"]

        answer = assistant.generate_answer(query, context=context)

        # Check for image path and incorporate it into the answer
        image_path = metadata.get("image_path")
        if image_path:
            answer += f"\n\nHere's a diagram to help you understand:\n![Diagram]({image_path})"
    else:
        logging.info("No relevant information found in knowledge base, trying Gemini API.")
        answer = assistant.generate_answer(query)

    if not answer:
        return "I'm sorry, I cannot answer the question", None

    pdf_file_name = f"answer_{int(time.time())}.pdf"
    create_pdf(query, answer, pdf_file_name)
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
        logging.error(f"Error reading PDF : {e}")
        return None

nlp = load_nlp_model()
#knowledge_base = load_knowledge_base('knowledge_base.json') # load this via KnowledgeBaseManager now.
assistant = Gemini_Assistant()
kb_manager = KnowledgeBaseManager() # Load knowledge base and build FAISS index

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_request():
    try:
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
                            kb_manager.knowledge_base["PDF"] = kb_manager.knowledge_base.get("PDF",[]) + [{"pdf_filename": filename, "content": pdf_text}] # Add to knowledge base with PDF
                            # Save knowledge_base with the new entry
                            with open('knowledge_base.json', 'w') as f:
                                json.dump(kb_manager.knowledge_base, f, indent=4)
                            kb_manager.build_index() # Rebuild the FAISS index

                    if os.path.exists(file_path):
                       os.remove(file_path) # Delete the file after processing
                else:
                    return jsonify({'answer' : "Error reading the file"}), 400

            answer, pdf_file = process_user_query(nlp, assistant, kb_manager, query=None) # Pass kb_manager

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
                  answer, pdf_file = process_user_query(nlp, assistant, kb_manager, query=text) # Pass kb_manager
                else:
                     return jsonify({'answer' : "Could not understand speech"}), 400
            elif query:
                answer, pdf_file = process_user_query(nlp, assistant, kb_manager, query=query) # Pass kb_manager
            else:
                return jsonify({'answer' : "Could not understand the request"}), 400
        else:
           return jsonify({'answer' : "Invalid request"}), 400

        return jsonify({'answer': answer, "pdf_file": pdf_file })

    except Exception as e:
        logging.exception("An error occurred during the processing of the request.")
        return jsonify({'answer': f"An error occurred: {str(e)}"}, 500)

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
    #Initialize the Knowledge Base and FAISS Index on startup.
    with app.app_context():
        kb_manager = KnowledgeBaseManager()
    app.run(debug=True)