import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Setup LLM (Groq with Llama)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create QA chain using modern LCEL approach
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    | load_llm()
    | StrOutputParser()
)

# Now invoke with a single query
user_query = input("Write Query Here: ")

# Get source documents separately
source_documents = retriever.invoke(user_query)

# Get the answer
response_result = qa_chain.invoke(user_query)

print("RESULT: ", response_result)
print("\nSOURCE DOCUMENTS: ")
for i, doc in enumerate(source_documents, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:300] + "...")