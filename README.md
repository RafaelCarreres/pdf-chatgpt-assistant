# pdf-chatgpt-assistant
This repository contains a Python script that integrates ChatGPT (using OpenAI API) with PDF files. It allows users to ask questions about the content of a selected PDF file, and the ChatGPT model will generate responses based on the summarized context of the document.

## Features
- Extracts text from a PDF file.
- Summarizes the extracted text using the T5 summarization model from the Hugging Face Transformers library.
- Allows users to ask questions about the content of the PDF file.
- Generates responses to the questions using OpenAI's ChatGPT (text-davinci-003).
- Provides a simple graphical user interface (GUI) using Tkinter.
## Requirements
- Python 3.6 or higher
- PyPDF2
- spaCy
- openai
- transformers
- Tkinter
## Installation
1. Clone this repository:
    - git clone https://github.com/RafaelCarreres/pdf-chatgpt-assistant.git
2. Install the required packages:
    - pip install -r requirements.txt
3. Set the environment variable API_KEY with your OpenAI API key:
    - export API_KEY="your-api-key"
## Usage
1. Run the script:
   - python main.py
2. Click on the "Select PDF" button to choose a PDF file from your computer.
3. Type your question in the input dialog that appears and press Enter. The ChatGPT response will be displayed in the GUI.
4. To ask another question, repeat step 3.
5. To exit the application, type 'exit' in the input dialog or close the GUI window.
## Limitations
- The T5 summarization model used in this script has a token limit of 5120 tokens. If the PDF file has more tokens than this limit, the context might be truncated and some information may be lost.
- The ChatGPT model has a maximum token limit of 4096 tokens for input. If the summarized context and the question combined exceed this limit, the script will truncate the context to fit within the limit.
