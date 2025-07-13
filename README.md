# AI Virtual Tutor - RAG-Powered Educational Assistant

A comprehensive virtual tutoring system that combines natural language processing, speech recognition, and generative AI to provide personalized educational assistance. The system can handle text queries, voice input, image processing, and PDF document analysis.

## 🚀 Features

### Core Capabilities
- **RAG (Retrieval-Augmented Generation)**: Advanced semantic search using FAISS vector database and Sentence Transformers for context-aware responses
- **Multi-modal Input Support**: Text, voice, image, and PDF document processing
- **AI-Powered Responses**: Integration with Google's Gemini Pro for intelligent answers
- **Knowledge Base Management**: FAISS-based vector search for efficient content retrieval
- **Speech Recognition**: Real-time voice-to-text conversion using Google Speech Recognition
- **OCR Processing**: Text extraction from images using Tesseract
- **PDF Generation**: Automatic creation of answer documents in PDF format
- **Image Generation**: AI-generated visualizations using Stable Diffusion

### Educational Features
- **Subject Coverage**: Math, Physics, Chemistry, AI & Robotics, and more
- **Context-Aware Responses**: Leverages knowledge base for relevant answers using semantic similarity
- **Visual Learning**: Automatic image generation for complex concepts
- **Document Analysis**: Extract and learn from uploaded PDF materials
- **Conversation Memory**: Maintains context across multiple interactions
- **Dynamic Knowledge Base**: Automatically indexes new PDF content and rebuilds vector embeddings

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **AI/ML**: 
  - Google Gemini Pro (Language Model)
  - spaCy (NLP processing)
  - Sentence Transformers (Embeddings for RAG)
  - FAISS (Vector database for semantic search)
  - Stable Diffusion (Image generation)
  - RAG Architecture (Retrieval-Augmented Generation)
- **Speech Processing**: Google Speech Recognition
- **OCR**: Tesseract
- **Document Processing**: PyPDF2, ReportLab
- **Frontend**: HTML, CSS, JavaScript

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini Pro
- Tesseract OCR (for image text extraction)
- CUDA-compatible GPU (optional, for faster image generation)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd virtual-tutor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv vt_env
   source vt_env/bin/activate  # On Windows: vt_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Install Tesseract OCR**
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **macOS:**
   ```bash
   brew install tesseract
   ```
   
   **Windows:**
   Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

6. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🎯 Usage

### Starting the Application

1. **Run the main application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

### Available Applications

The project includes three different application versions with varying RAG capabilities:

- **`app.py`**: Full-featured version with advanced RAG, image generation, and dynamic knowledge base updates
- **`application.py`**: Core version with RAG implementation but without image generation
- **`multi_app.py`**: Simplified version with basic keyword-based search and limited RAG features

### Using the Virtual Tutor

1. **Text Queries**: Type your question in the text input field
2. **Voice Input**: Click the "Voice" button and speak your question
3. **Image Upload**: Upload images containing text or diagrams for analysis
4. **PDF Upload**: Upload educational PDFs to add to the knowledge base
5. **Download Answers**: Generated answers are automatically saved as PDF files

### Supported File Types

- **Images**: PNG, JPG, JPEG
- **Documents**: PDF, DOCX
- **Audio**: WAV (for voice input)

## 📁 Project Structure

```
virtual-tutor/
├── app.py                 # Main application with full features
├── application.py         # Core application without image generation
├── multi_app.py          # Simplified application
├── knowledge_base.json   # Knowledge base configuration
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── templates/
│   └── index.html       # Web interface
├── static/              # Static assets
├── uploads/             # Temporary file uploads
├── results/             # Generated PDF answers
└── input/              # Sample educational materials
    ├── maths_ncert.pdf
    ├── mathsEM.pdf
    ├── science_ncert.pdf
    └── logo.png
```

## 🔧 Configuration

### RAG Architecture

The system implements a sophisticated RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Indexing**: Educational content is converted to vector embeddings using Sentence Transformers
2. **Semantic Search**: FAISS vector database enables fast similarity search across knowledge base
3. **Context Retrieval**: Most relevant content is retrieved based on semantic similarity
4. **Answer Generation**: Gemini Pro generates responses using retrieved context as grounding
5. **Dynamic Updates**: New PDF content is automatically indexed and embeddings are rebuilt

### Knowledge Base Setup

The `knowledge_base.json` file contains structured educational content:

```json
{
    "Math": [],
    "Physics": [],
    "Chemistry": [],
    "AI & Robotics": [
        {
            "content": "Educational content here",
            "keywords": ["relevant", "keywords"],
            "image_path": "optional_image_path",
            "generate_image": "optional_image_prompt"
        }
    ],
    "PDF": [
        {
            "pdf_filename": "path/to/pdf",
            "content": "extracted_text_content"
        }
    ]
}
```

### Environment Variables

- `GOOGLE_API_KEY`: Required for Gemini Pro API access
- Additional configuration can be added to `.env` file

## 🎨 Customization

### Adding New Subjects

1. Add new categories to `knowledge_base.json`
2. Include relevant content with proper structure
3. Restart the application to rebuild the FAISS index

### Modifying AI Models

- Change Gemini model in `Gemini_Assistant` class
- Update embedding model in `KnowledgeBaseManager` (affects RAG performance)
- Modify Stable Diffusion model in `ImageGenerator`
- Adjust FAISS index parameters for different vector dimensions

### Styling the Interface

Edit `templates/index.html` to customize the web interface appearance and functionality.

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env` file
2. **Tesseract Not Found**: Verify Tesseract installation and path
3. **spaCy Model Missing**: Run `python -m spacy download en_core_web_sm`
4. **CUDA Issues**: Set device to "cpu" in `ImageGenerator` if GPU unavailable

### Performance Optimization

- Use GPU for faster image generation and embedding computation
- Adjust FAISS index parameters for larger knowledge bases
- Optimize image generation steps based on hardware capabilities
- Fine-tune embedding model for better semantic search accuracy
- Consider using FAISS-GPU for large-scale vector operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code comments for implementation details

## 🔮 Future Enhancements

- Multi-language support with multilingual embeddings
- Advanced conversation management with conversation memory
- Integration with learning management systems
- Real-time collaboration features
- Mobile application
- Advanced analytics and progress tracking
- Hybrid search combining keyword and semantic matching
- Custom fine-tuning of embedding models for educational content
- Multi-modal RAG with image and text understanding
