#!/bin/bash

# Virtual Tutor - Quick Start Script
# This script helps you set up and run the Virtual Tutor application

set -e  # Exit on any error

echo "🚀 Virtual Teaching Assistant - Quick Start Setup"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_header "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_header "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_status "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_header "Creating virtual environment..."
    if [ ! -d "vt_env" ]; then
        python3 -m venv vt_env
        print_status "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_header "Activating virtual environment..."
    source vt_env/bin/activate
    print_status "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_status "Dependencies installed successfully"
}

# Install spaCy model
install_spacy_model() {
    print_header "Installing spaCy language model..."
    python -m spacy download en_core_web_sm
    print_status "spaCy model installed"
}

# Check Tesseract installation
check_tesseract() {
    print_header "Checking Tesseract OCR installation..."
    if command -v tesseract &> /dev/null; then
        TESSERACT_VERSION=$(tesseract --version | head -n1 | cut -d' ' -f2)
        print_status "Tesseract $TESSERACT_VERSION found"
    else
        print_warning "Tesseract OCR is not installed."
        echo "Please install Tesseract:"
        echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
        echo "  macOS: brew install tesseract"
        echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    fi
}

# Create .env file
create_env_file() {
    print_header "Setting up environment variables..."
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Virtual Tutor Environment Variables
# Replace with your actual Google API key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Configure other settings
# DEBUG=True
# HOST=0.0.0.0
# PORT=5000
EOF
        print_status ".env file created"
        print_warning "Please edit .env file and add your Google API key"
    else
        print_warning ".env file already exists"
    fi
}

# Create necessary directories
create_directories() {
    print_header "Creating necessary directories..."
    mkdir -p uploads results static
    print_status "Directories created"
}

# Check if Google API key is set
check_api_key() {
    print_header "Checking Google API key..."
    if [ -f ".env" ]; then
        if grep -q "your_google_api_key_here" .env; then
            print_warning "Please update your Google API key in the .env file"
            echo "You can get a free API key from: https://makersuite.google.com/app/apikey"
        else
            print_status "Google API key appears to be configured"
        fi
    else
        print_error ".env file not found. Please run setup again."
        exit 1
    fi
}

# Run the application
run_application() {
    print_header "Starting Virtual Tutor application..."
    echo ""
    echo "🌐 The application will be available at: http://localhost:5000"
    echo "📝 Press Ctrl+C to stop the application"
    echo ""
    python app.py
}

# Main setup function
main_setup() {
    check_python
    check_pip
    create_venv
    activate_venv
    install_dependencies
    install_spacy_model
    check_tesseract
    create_env_file
    create_directories
    check_api_key
    
    echo ""
    print_status "Setup completed successfully! 🎉"
    echo ""
    echo "Next steps:"
    echo "1. Edit .env file and add your Google API key"
    echo "2. Run: ./quick_start.sh run"
    echo ""
}

# Main function
main() {
    case "${1:-setup}" in
        "setup")
            main_setup
            ;;
        "run")
            activate_venv
            check_api_key
            run_application
            ;;
        "install")
            activate_venv
            install_dependencies
            install_spacy_model
            ;;
        "check")
            check_python
            check_pip
            check_tesseract
            check_api_key
            ;;
        "help"|"-h"|"--help")
            echo "Virtual Teaching Assistant - Quick Start Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  setup   - Complete setup (default)"
            echo "  run     - Run the application"
            echo "  install - Install dependencies only"
            echo "  check   - Check system requirements"
            echo "  help    - Show this help message"
            echo ""
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 