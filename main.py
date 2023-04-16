import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
import PyPDF2
import spacy
import openai
from transformers import pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # It has a limit of 1024 tokens
# from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", tokenizer="sshleifer/distilbart-cnn-12-6")

# It can handle 5120 tokens
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Get the API key from the environment variable
api_key = os.environ.get('API_KEY')

# Set the OpenAI API key
openai.api_key = api_key

# Load spaCy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

def select_pdf():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        process_pdf(file_path)

def display_response(response):
    response_text.delete(1.0, tk.END)
    response_text.insert(tk.END, response)

def extract_text_from_pdf(file_path):
    pdf_file = open(file_path, "rb")
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

    pdf_file.close()
    return text

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def query_chatgpt(prompt, context, max_length=4096):
    # Summarize the context to ensure it fits within the token limit
    input_length = len(context.split())
    summary_length = min(max(input_length // 2, 50), 1500)
    summarized_context = summarizer(context, min_length=50, max_length=summary_length)[0]["summary_text"]

    # Prepend context to the prompt
    full_prompt = f"{summarized_context}\n\n{prompt}"

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=full_prompt,
            temperature=0.5,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            stop=None,
        )

        return response.choices[0].text.strip()
    except openai.OpenAIError as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request."

def process_pdf(file_path):
    raw_text = extract_text_from_pdf(file_path)
    context = summarizer(raw_text, truncation=True, max_length=4096)[0]["summary_text"]  # Increase max_length to 4096

    # Truncate the context to the first 8192 tokens (or any other limit)
    tokenized_context = tokenizer(context, return_tensors="pt", truncation=True, max_length=8192)  # Increase max_length to 8192
    truncated_context = tokenizer.decode(tokenized_context["input_ids"][0])

    while True:  # Start a loop to ask multiple questions
        prompt = simpledialog.askstring("Question", "Please enter your question (or type 'exit' to stop):")

        if prompt and prompt.lower() != "exit":  # Check if the user provided a question or wants to exit
            response = query_chatgpt(prompt, truncated_context, max_length=200)  # Increase max_length for longer responses
            display_response(response)
        else:
            break  # Exit the loop if the user typed 'exit'

def create_gui():
    global response_text

    root = tk.Tk()
    root.title("ChatGPT-PDF Integration")

    select_button = tk.Button(root, text="Select PDF", command=select_pdf)
    select_button.pack()

    response_label = tk.Label(root, text="ChatGPT Response:")
    response_label.pack()

    response_text = tk.Text(root, wrap=tk.WORD)
    response_text.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui()
