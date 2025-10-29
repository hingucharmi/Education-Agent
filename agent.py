import os
from dotenv import load_dotenv
from typing import List, Dict, Any
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
LLM_MODEL = "gpt-4o-mini" # Or gpt-3.5-turbo

class ExplanationAgent:
    """
    Handles generating explanations based on context, with an adjustable persona and complexity.
    This acts as the "Pedagogy & Delivery" agent.
    """
    def __init__(self, llm_model: ChatOpenAI, initial_persona: str = "friendly high school teacher", complexity: str = "medium"):
        self.llm = llm_model
        self.persona = initial_persona
        self.complexity = complexity # Can be 'low', 'medium', 'high'
        print(f"ExplanationAgent initialized. Persona: {self.persona}, Complexity: {self.complexity}")

    def update_settings(self, new_persona: str = None, new_complexity: str = None):
        """Allows the Evolution Agent to change the explanation style."""
        if new_persona:
            self.persona = new_persona
        if new_complexity:
            self.complexity = new_complexity
        print(f"ExplanationAgent settings updated. New Persona: {self.persona}, New Complexity: {self.complexity}")

    def get_explanation_prompt(self) -> str:
        """Dynamically generates the system prompt based on current settings."""
        tone_map = {
            "low": "very simple, using common words and basic analogies for a middle schooler.",
            "medium": "engaging, using analogies and examples suitable for a high school student.",
            "high": "detailed and precise, suitable for a college student, still with clarity."
        }
        
        return f"""
        You are an expert academic tutor with the persona of a **{self.persona}**.
        Your goal is to answer the user's question based ONLY on the provided context,
        in a {tone_map.get(self.complexity, 'engaging and easy-to-understand way')}.

        Follow these rules:
        1. Simplify and Relate: Break down complex concepts into analogies, real-world examples, or a short story.
        2. Tone: Be encouraging and supportive.
        3. Accuracy: If the context does not contain the answer, state that you couldn't find the information in the document.
        4. Conciseness: Keep explanations focused and to the point for the current complexity level.

        CONTEXT:
        {{context}}
        """

    def create_qa_chain(self, chat_history: List[BaseMessage]):
        """Creates the question-answering chain with the current explanation settings."""
        qa_system_prompt = self.get_explanation_prompt()
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, qa_prompt)


class EvolutionAgent:
    """
    Assesses student understanding and directs the ExplanationAgent to adapt.
    This acts as the "Assessment & Iteration" agent.
    """
    def __init__(self, llm_model: Any):
        self.llm = llm_model
        print("EvolutionAgent initialized.")

    def assess_and_recommend_action(self, student_question: str, tutor_explanation: str, student_feedback: str) -> Dict[str, Any]:
        """
        Assesses if the student understood and recommends a new persona/complexity
        or if further explanation is needed.
        """
        print(f"-> EvolutionAgent: Assessing understanding from student feedback...")

        assessment_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an intelligent assessment system. Your task is to analyze a student's feedback "
            "after receiving an explanation from a tutor. "
            "Determine if the student clearly understood the topic. "
            "If they understood, suggest an action to 'advance' (e.g., maintain complexity, or subtly increase it for the next topic). "
            "If they did NOT understand or expressed confusion, suggest an action to 're-explain' (e.g., lower complexity, change persona to 'super simplifier').\n\n"
            "Provide your response in a JSON format:\n"
            "```json\n"
            "{{ \"understood\": true/false,\n  \"recommended_action\": \"advance\"/\"re-explain\",\n"
            "  \"new_persona\": \"[optional new persona]\",\n"
            "  \"new_complexity\": \"[optional new complexity: low, medium, high]\",\n"
            "  \"assessment_notes\": \"[brief notes on assessment]\" }}\n```\n"
            "Strictly adhere to the JSON format. If no change is needed, omit 'new_persona' or 'new_complexity'."),
            ("user",
             f"Tutor's Explanation: {tutor_explanation}\n"
             f"Student's Feedback: {student_feedback}\n"
             f"Original Question: {student_question}")
        ])

        try:
           # 💡 CRITICAL FIX: Invoke the template to get the prompt value
            prompt_value = assessment_prompt.invoke({})

            # Pass the correctly formatted prompt value to the LLM
            response_str = self.llm.invoke(prompt_value).content
            
            # Attempt to parse JSON. Sometimes LLMs add extra text.
            if "```json" in response_str:
                response_str = response_str.split("```json")[1].split("```")[0]
            
            assessment = json.loads(response_str)
            print(f"-> EvolutionAgent: Assessment result: {assessment}")
            return assessment
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM: {e}")
            print(f"LLM raw response: {response_str}")
            # Fallback to a default action
            return {
                "understood": False,
                "recommended_action": "re-explain",
                "new_complexity": "low",
                "assessment_notes": "JSON decode error, assuming confusion."
            }
        except Exception as e:
            print(f"An unexpected error in EvolutionAgent: {e}")
            return {
                "understood": False,
                "recommended_action": "re-explain",
                "new_complexity": "low",
                "assessment_notes": "Unexpected error, assuming confusion."
            }


class AcademicTutorAgent:
    """A conversational RAG agent designed to simplify PDF content for students."""

    def __init__(self):
        self.pdf_path: Path = None # Will be set by student upload
        self.chat_history: List[BaseMessage] = []
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7) # Slightly higher temp for creativity
        self.vector_store = None
        self.retriever = None

        # Initialize the sub-agents
        self.explanation_agent = ExplanationAgent(self.llm)
        self.evolution_agent = EvolutionAgent(self.llm)
        
        self.current_question = "" # To store the last question for re-explanation

    # --- Input Agent Functionality ---
    def ingest_student_pdf(self, pdf_path_str: str):
        """Allows the student to upload/specify their PDF."""
        self.pdf_path = Path(pdf_path_str)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Error: PDF file not found at '{self.pdf_path}'")
        
        print(f"-> Processing document: {self.pdf_path.name}...")

        loader = PyPDFLoader(str(self.pdf_path))
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"-> Document split into {len(chunks)} chunks.")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma.from_documents(chunks, embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        print("-> PDF ingested and RAG components ready.")
        self.chat_history = [] # Reset chat history for new document

    # --- RAG Chain Construction (incorporating Explanation Agent) ---
    def _create_full_rag_chain(self):
        """
        Builds the History-Aware RAG chain using the current settings
        of the Explanation Agent.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Please ingest a PDF first.")

        # History-aware retriever (Remains constant)
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question "
                     "which might reference context in the chat history, "
                     "formulate a standalone question which can be understood "
                     "without the chat history. Do NOT answer the question, "
                     "just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, condense_question_prompt
        )

        # Question-answer chain (Uses dynamic prompt from ExplanationAgent)
        question_answer_chain = self.explanation_agent.create_qa_chain(self.chat_history)
        
        # Combine into full RAG chain
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def chat_loop(self):
        """The main console loop for continuous conversation."""
        print("\n" + "="*50)
        print(f"Academic Tutor: Interactive Session")
        print("Type 'upload' to load a new PDF, 'quit' or 'exit' to end.")
        print("="*50)
        
        # Initial PDF upload
        pdf_file = input("Please enter the path to the PDF document you want to learn from: ")
        try:
            self.ingest_student_pdf(pdf_file)
            rag_chain = self._create_full_rag_chain() # Initialize chain after PDF is loaded
            print(f"Tutor is ready to discuss: {self.pdf_path.name}")
        except FileNotFoundError as e:
            print(e)
            print("Please restart and provide a valid PDF path.")
            return
        except Exception as e:
            print(f"An error occurred during PDF ingestion: {e}")
            print("Please restart and try again.")
            return

        while True:
            try:
                user_input = input("\nStudent: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("\nTutor: Goodbye! Keep learning and exploring! 🚀")
                    break
                elif user_input.lower() == "upload":
                    new_pdf_file = input("Enter path for the new PDF: ")
                    try:
                        self.ingest_student_pdf(new_pdf_file)
                        rag_chain = self._create_full_rag_chain() # Re-initialize chain
                        print(f"\nTutor is now ready to discuss: {self.pdf_path.name}")
                    except Exception as e:
                        print(f"Failed to load new PDF: {e}. Continuing with previous document.")
                    continue
                
                # Store the current question for potential re-explanation
                self.current_question = user_input
                
                print("Tutor is thinking...")
                response = rag_chain.invoke({
                    "chat_history": self.chat_history,
                    "input": user_input
                })
                tutor_explanation = response["answer"]

                print(f"\nTutor: {tutor_explanation}")

                # --- Evolution Agent interaction ---
                # Prompt student for feedback on understanding
                feedback = input("Student (Did you understand? Say 'yes', 'no', or elaborate): ")
                
                # If the student didn't understand, the Evolution Agent steps in
                if feedback.lower() in ["no", "not really", "confused", "can you explain again?"]:
                    print("Tutor: No worries! Let me try explaining that a different way.")
                    assessment = self.evolution_agent.assess_and_recommend_action(
                        student_question=self.current_question,
                        tutor_explanation=tutor_explanation,
                        student_feedback=feedback # Pass the negative feedback
                    )

                    # Update Explanation Agent based on assessment
                    if assessment["recommended_action"] == "re-explain":
                        new_complexity = assessment.get("new_complexity", "low")
                        new_persona = assessment.get("new_persona", "super simplifier") # Default to very simple on confusion
                        self.explanation_agent.update_settings(
                            new_persona=new_persona,
                            new_complexity=new_complexity
                        )
                        print(f"Tutor's approach updated: Persona='{self.explanation_agent.persona}', Complexity='{self.explanation_agent.complexity}'")
                        
                        # Re-ask the *same* question with the updated persona/complexity
                        print("\nTutor is re-explaining...")
                        # We need to re-create the chain with the new explanation agent settings
                        rag_chain = self._create_full_rag_chain() 
                        
                        re_explanation_response = rag_chain.invoke({
                            "chat_history": self.chat_history, # Use existing history
                            "input": self.current_question # Use the original question
                        })
                        tutor_explanation = re_explanation_response["answer"]
                        print(f"\nTutor (Re-explanation): {tutor_explanation}")
                        
                        # After re-explanation, store the new explanation in history
                        self.chat_history.append(HumanMessage(content=self.current_question))
                        self.chat_history.append(AIMessage(content=tutor_explanation))

                        # Optionally, ask for feedback again after re-explanation
                        # For simplicity, we'll let the loop continue and expect next user input
                        # In a more advanced system, you might have a nested feedback loop here.
                    else: # If assessment somehow suggests advance even on 'no' feedback, which shouldn't happen with current prompt
                        print("Tutor: Hmm, that's odd. Let's try to break it down. Tell me what part was confusing.")

                else: # Student understood, update chat history and potentially adapt for next question
                    assessment = self.evolution_agent.assess_and_recommend_action(
                        student_question=self.current_question,
                        tutor_explanation=tutor_explanation,
                        student_feedback=feedback # Pass positive feedback
                    )
                    if assessment["recommended_action"] == "advance":
                         # Gently adjust persona/complexity upwards if applicable, or keep stable
                        current_complexity = self.explanation_agent.complexity
                        if current_complexity == "low":
                            self.explanation_agent.update_settings(new_complexity="medium", new_persona="friendly high school teacher")
                        elif current_complexity == "medium" and assessment.get("new_complexity"):
                            self.explanation_agent.update_settings(new_complexity="high", new_persona="collegiate guide")
                        
                        print(f"Tutor's approach updated for next question: Persona='{self.explanation_agent.persona}', Complexity='{self.explanation_agent.complexity}'")
                        rag_chain = self._create_full_rag_chain() # Update chain with potentially new settings
                        
                    # Always update chat history after a successful exchange
                    self.chat_history.append(HumanMessage(content=user_input))
                    self.chat_history.append(AIMessage(content=tutor_explanation))


            except Exception as e:
                print(f"\nAn error occurred: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                print("Please try your question again.")
                
if __name__ == "__main__":
    import json # For parsing LLM output in EvolutionAgent
    if "OPENAI_API_KEY" not in os.environ:
        print("FATAL ERROR: The OPENAI_API_KEY environment variable is not set.")
    else:
        try:
            agent = AcademicTutorAgent()
            agent.chat_loop()

        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"An unexpected error occurred during setup: {e}")