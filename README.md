## Prerequisites
++ Python 3.9+
++ Git
++ Text Editor

## Download Ollama 
```
https://ollama.com/download
```

#### MAC
<img width="535" alt="Screenshot 2024-07-02 at 8 53 49 PM" src="https://github.com/rbhogara/Test/assets/126253116/9ff5a7d6-d5f3-48ee-8a4c-e385c2651180"><br>
<img width="396" alt="Screenshot 2024-07-02 at 8 54 42 PM" src="https://github.com/rbhogara/Test/assets/126253116/80934283-68b0-4087-a8e5-eb6c594551e8">


#### Windows
<img width="548" alt="Screenshot 2024-07-02 at 8 58 25 PM" src="https://github.com/rbhogara/Test/assets/126253116/d91acd86-f133-4e4b-84ab-f25f0acfeea4"><br>

#### Linux
<img width="580" alt="Screenshot 2024-07-02 at 8 58 14 PM" src="https://github.com/rbhogara/Test/assets/126253116/96f330be-e72d-4406-af80-e4e34caacce7"><br>
```
curl -fsSL https://ollama.com/install.sh | sh
```

## Dependencies

To Create virtual environment (Optional):

```
python3 -m venv talk-to-docs
source talk-to-docs/bin/activate
```

To run this application, you need to have the following dependencies installed:

Ollama Dependencies -
```
ollama pull llama3
ollama pull nomic-embed-text
```
Python Dependencies - 
```
pip install -r requirements.txt
```

## Usage

1. Run the application using the following command:
```
streamlit run app.py
```
2. Upload your documents (PDF, DOCX, PPTX) by clicking the "Upload files" button.

3. Once the documents are uploaded, click the "Generate Embeddings" button to process the documents and create the necessary embeddings.

4. Enter your question in the text input field and click the "Query Documents" button to get the answer based on the uploaded documents.


## Resources
This application uses the following libraries and tools:

- [Streamlit](https://streamlit.io/) for the web application framework.
- [LangChain](https://langchain.com/) for the language model and document processing.
- [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF file processing.
- [python-docx](https://python-docx.readthedocs.io/) for DOCX file processing.
- [python-pptx](https://python-pptx.readthedocs.io/) for PPTX file processing.
- [Chroma](https://www.trychroma.com/) for the vector database.
- [Ollama](https://www.anthropic.com/models) for the language model.
