import os
from dotenv import load_dotenv
from typing import List
from pathlib import Path

# LangChain Imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Load environment variables
load_dotenv()
LLM_MODEL = "gpt-4o-mini" 

class AcademicTutorAgent:
    """A conversational RAG agent designed to simplify PDF content for students."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.chat_history: List[BaseMessage] = []
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5)
        self.rag_chain = self._initialize_chain()

    def _get_retriever(self):
        """Loads PDF, chunks it, creates embeddings, and returns a Retriever."""
        print(f"-> Processing document: {self.pdf_path.name}...")

        # 1. Load and Split
        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"-> Document split into {len(chunks)} chunks.")

        # 2. Embed and Store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma.from_documents(chunks, embeddings) 
        
        return vector_store.as_retriever(search_kwargs={"k": 3}) 

    def _create_history_aware_chain(self, retriever):
        """Builds the History-Aware RAG chain."""
        
        # --- A. Prompt to Rephrase Question Based on History ---
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                     "which might reference context in the chat history, "
                     "formulate a standalone question which can be understood "
                     "without the chat history. Do NOT answer the question, "
                     "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        # This chain handles rephrasing the question and retrieving documents.
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, condense_question_prompt
        )

        # --- B. Prompt to Generate the Final Answer ---
        qa_system_prompt = """
        You are an expert academic tutor and creative storyteller. 
        Your goal is to answer the user's question based ONLY on the provided context, 
        but in a fun, engaging, and easy-to-understand way.

        Follow these rules:
        1. Simplify and Relate: Break down complex concepts into analogies, real-world examples, or a short story.
        2. Tone: Be encouraging and use language for a high school student.
        3. Accuracy: If the context does not contain the answer, state that you couldn't find the information in the document.

        CONTEXT:
        {context}
        """
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        # This chain takes the question and retrieved documents and generates an answer.
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # --- C. Final Conversational RAG Chain ---
        # This combines the two chains above into a complete workflow.
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def _initialize_chain(self):
        """Initializes the retriever and the RAG chain."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Error: PDF file not found at '{self.pdf_path}'")
        
        retriever = self._get_retriever()
        print("-> RAG chain initialized. Ready for chat.")
        return self._create_history_aware_chain(retriever)

    def chat_loop(self):
        """The main console loop for continuous conversation."""
        print("\n" + "="*50)
        print(f"Academic Tutor: Chatting with '{self.pdf_path.name}'")
        print("Type 'quit' or 'exit' to end the session.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nStudent: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("\nTutor: Goodbye! Keep learning and exploring! 🚀")
                    break
                
                print("Tutor is thinking...")
                
                # Invoke the chain with the correct input structure
                response = self.rag_chain.invoke({
                    "chat_history": self.chat_history,
                    "input": user_input
                })
                tutor_response = response["answer"]

                # Update chat history for the next turn
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=tutor_response))
                
                print(f"\nTutor: {tutor_response}")

            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please try your question again.")
                
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL ERROR: The OPENAI_API_KEY environment variable is not set.")
    else:
        pdf_file = input("Enter the path to your PDF document: ")
        
        try:
            agent = AcademicTutorAgent(pdf_file)
            agent.chat_loop()

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred during setup: {e}")