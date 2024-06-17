import streamlit as st
import io
import PyPDF2
import docx
import pptx
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
import time
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    with io.BytesIO(uploaded_file.read()) as f:
        if uploaded_file.name.endswith('.pdf'):
            return extract_text_from_pdf(f)
        elif uploaded_file.name.endswith('.docx'):
            return extract_text_from_docx(f)
        elif uploaded_file.name.endswith('.pptx'):
            return extract_text_from_pptx(f)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return None

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF file: {file.name}")
        st.error(str(e))
        return None

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = "".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error processing DOCX file: {file.name}")
        st.error(str(e))
        return None

def extract_text_from_pptx(file):
    try:
        presentation = pptx.Presentation(file)
        text = ""
        for slide in presentation.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text += shape.text
        return text
    except Exception as e:
        st.error(f"Error processing PPTX file: {file.name}")
        st.error(str(e))
        return None

def process_input(uploaded_files):
    docs_list = [extract_text_from_file(file) for file in uploaded_files if extract_text_from_file(file)]
    docs_list = [doc for doc in docs_list if doc]  # Remove None values
    
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(" ".join(docs_list))

    Document = namedtuple('Document', ['page_content', 'metadata'])
    documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]

    # Convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="rag-chroma",
        embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text'),
    )
    return vectorstore.as_retriever()

def question(question, retriever):
    model_local = Ollama(model="mistral")

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

def process_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error processing CSV file: {uploaded_file.name}")
        st.error(str(e))
        return None

def plot_data(df):
    st.write("Select columns to plot:")
    columns = st.multiselect("Columns", df.columns)
    if st.button('Plot'):
        for column in columns:
            fig, ax = plt.subplots()
            if df[column].dtype == 'object':
                ax.bar(df[column].value_counts().index, df[column].value_counts().values)
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.set_title(f'Categorical Data: {column}')
            elif df[column].dtype in ['int64', 'float64']:
                ax.hist(df[column], bins=50)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Numerical Data: {column}')
            else:
                st.warning("Unsupported data type.")
            st.pyplot(fig)

def main():
    st.title("DocuTalk")
    st.write("Upload files and enter a question to query the documents.")

    uploaded_files = st.file_uploader("Upload files", type=["pdf", "docx", "pptx", "csv"], accept_multiple_files=True)

    if uploaded_files:
        csv_files = [file for file in uploaded_files if file.name.endswith('.csv')]
        if csv_files:
            st.write("CSV file detected. Plotting data...")
            for file in csv_files:
                df = process_csv(file)
                if df is not None:
                    plot_data(df)
        else:
            if st.button('Generate Embeddings'):
                with st.spinner('Generating Embeddings...'):
                    retriever = process_input(uploaded_files)
                    st.success('Embeddings generated successfully!')

            question_input = st.text_input("Enter your question:")

            if st.button('Query Documents'):
                if not question_input:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner('Processing...'):
                        retriever = process_input(uploaded_files)
                        answer = question(question_input, retriever)
                        st.text_area("Answer", value=answer, height=300, disabled=True)
    else:
        st.warning("Please upload at least one file.")

if __name__ == "__main__":
    main()


