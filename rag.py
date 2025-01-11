import streamlit as st
from dotenv import load_dotenv #to load the env variables i.e the API keys
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI #faced rate limit error
from langchain.memory import ConversationBufferMemory
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    #when we enter pdfs in the sidebar this function is used to display the raw text from the pdfs which is further used to create chunks
    text = ""
    #looping though each page of the pdf and adding it to the text variable
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    #this function is used to split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator = "\n", #used to split the text into single line breaks
        chunk_size = 1000, #size of each chunk
        chunk_overlap = 200, #overlap takes a few characters from previous chunks to maintain context
        length_function = len #used to calculate the length of the text
    )
    chunks = text_splitter.split_text(text) #returns a list of 1000 character chunks with an overlap of 200 characters
    return chunks

def get_vectorstore(text_chunks):
    #this function is used to create a vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #this is the embedding model 
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)   #FAISS is a used for similarity search which maintains relevance of response
    return vectorstore

def get_conversation_chain(vectorstore):
    #this function is used to create a conversation chain which helps to answer follow up questions with the same context
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Add source documents to the response
    #conversation chain helps to backtrack the context of reponses after a follow up question
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True  # This will return source documents
    )
    return conversation_chain

def handle_userinput(user_input):
    # Check if conversation chain exists
    if st.session_state.conversation is None:
        st.error("Please upload and process a document first!")
        return
        
    #this function is used to handle the user input
    response = st.session_state.conversation({"question": user_input})
    st.session_state.chat_history = response["chat_history"]

    #looping through the chat history and displaying the messages
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html = True)

def get_document_text(docs):

    text = ""
    for doc in docs:
        file_type = doc.name.split('.')[-1].lower()
        if file_type == 'pdf':
            text += get_pdf_text([doc])
        elif file_type == 'txt':
            text += str(doc.read(), 'utf-8')
        elif file_type in ['docx', 'doc']:
            text += get_docx_text(doc)
    return text

def get_docx_text(doc):

    from docx import Document
    document = Document(doc)
    return " ".join([paragraph.text for paragraph in document.paragraphs])

def add_download_button():
    #this function is used to download the chat history
    if st.session_state.chat_history:
        chat_text = ""
        for message in st.session_state.chat_history:
            chat_text += f"{'Bot' if message.type=='ai' else 'User'}: {message.content}\n\n"
        
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

def add_clear_button():
    #this function is used to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = None
        st.session_state.conversation = None
        st.experimental_rerun()

def process_documents(pdf_docs):
    #this function is used to process the documents
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process text extraction
    status_text.text("Extracting text from documents...")
    progress_bar.progress(25)
    raw_text = get_pdf_text(pdf_docs)
    
    # Process text chunking
    status_text.text("Splitting text into chunks...")
    progress_bar.progress(50)
    text_chunks = get_text_chunks(raw_text)
    
    # Create vectorstore
    status_text.text("Creating vector embeddings...")
    progress_bar.progress(75)
    vectorstore = get_vectorstore(text_chunks)
    
    # Final setup
    status_text.text("Setting up conversation chain...")
    progress_bar.progress(100)
    st.session_state.conversation = get_conversation_chain(vectorstore)
    
    status_text.text("Ready to answer questions!")
    return vectorstore

def add_sidebar_settings():
    #this function is used to add the sidebar settings
    with st.sidebar:
        st.subheader("System Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        chunk_size = st.slider("Chunk Size", 500, 2500, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        return {"temperature": temperature, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}

def main():
    load_dotenv()
    st.set_page_config(page_title = "Question Answering System", page_icon = ":books:")

    st.write(css, unsafe_allow_html = True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Question Answering System")

    user_input = st.text_input("Ask a question:")
    if user_input:
        handle_userinput(user_input)

    with st.sidebar:
        st.subheader("Enter Documents to be used for the Question Answering System : ")
        pdf_docs = st.file_uploader("Upload Documents and click on Process", type = ["pdf", "docx", "txt"], accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get the text from the documents
                raw_text = get_pdf_text(pdf_docs)
                #split the text into chunks
                text_chunks = get_text_chunks(raw_text)
                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now ask questions about your documents.")

if __name__ == "__main__":
    main()

