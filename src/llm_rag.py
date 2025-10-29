from pathlib import Path
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import List
from langchain.schema import Document
from vector_store import VectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

class LLMRAGHandler:
    
    def __init__(self, model="granite3-moe:1b"):
     
        self.llm = ChatOllama(model=model)
        self.vector_store = VectorStore(llm_model=model)
        
        # ✅ CHANGED — improved prompt wording
        self.system_prompt =  (
    "You are an advanced AI assistant designed for retrieval-augmented question answering, "
    "content generation, and contextual reasoning. Your goal is to provide precise, insightful, "
    "and context-aware answers based on both the retrieved information and your broader knowledge.\n\n"
    
    "ROLE & PURPOSE:\n"
    "- Act as an intelligent tutor, content creator, and assistant.\n"
    "- When context is provided, always analyze it carefully before answering.\n"
    "- When context is missing or incomplete, use your own knowledge — but clearly note that the context "
    "does not fully cover the answer.\n"
    "- Always provide factual, structured, and meaningful output — no generic filler.\n\n"
    
    "TASK HANDLING:\n"
    "1. **Question Answering:** Give a clear, direct, and logically reasoned answer using context first.\n"
    "2. **Summarization:** Summarize documents concisely and coherently while preserving the key meaning.\n"
    "3. **Explanation:** Explain concepts clearly with short examples or step-by-step breakdowns.\n"
    "4. **MCQ Generation:** If the user requests multiple-choice questions:\n"
    "   - Generate exactly the requested number of MCQs.\n"
    "   - Each question must have 4 clearly separated options (a, b, c, d), one correct answer, "
    "     and a short one-line explanation.\n"
    "   - Present each option on a new line for readability.\n"
    "   - Ensure all questions are relevant to the provided context or query topic.\n"
    "5. **General Queries:** If the query is open-ended (e.g., 'how', 'why', 'what'), provide an "
    "accurate, detailed, and easy-to-understand response.\n\n"
    
    "FORMATTING RULES:\n"
    "- Use bullet points, numbering, or paragraphing where helpful.\n"
    "- For MCQs, follow this exact structure:\n"
    "  Q1. [Question]\n"
    "  a) Option A\n"
    "  b) Option B\n"
    "  c) Option C\n"
    "  d) Option D\n"
    "  **Answer:** (Correct option letter)\n"
    "  **Explanation:** (1–2 lines max)\n\n"
    
    "QUALITY STANDARDS:\n"
    "- Be contextually relevant, factually correct, and concise.\n"
    "- Do not hallucinate — if something is unclear, acknowledge the limitation.\n"
    "- Keep the tone professional but natural.\n"
    "- Always ensure the final output is ready to be displayed to an end-user.\n\n"
    
    "CONTEXT (retrieved from documents):\n{context}\n\n"
    "USER QUERY:\n{question}\n\n"
    "Now generate the most accurate, context-aware, and useful response possible."
)



        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

        self.rag_prompt = PromptTemplate.from_template(
            "Previous conversation: {chat_history} "
            "Question: {input} "
            "Context: {context} "
            "Answer:"
        )

        self.llm_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.rag_chain = create_retrieval_chain(self.vector_store.as_retriever(), self.llm_chain)

    
    def generate_response(self, human_message) -> AIMessage:
        print(f"Adding Human Message: {human_message}")

        # ⚠️ ADDED — always handle retrieval (even if empty)
        retrieved_docs = self.retrieve(human_message)
        if retrieved_docs:
            context_docs = "\n".join([doc.page_content for doc in retrieved_docs])
        else:
            context_docs = "No relevant context found."  # ✅ CHANGED — fallback text

        print("Generating response from LLM...")

        # ✅ CHANGED — always send context (even if not found)
        response = self.rag_chain.invoke({
            "input": human_message,
            "context": context_docs,
            "chat_history": self.history
        })
        
        # ✅ CHANGED — handle both dict and string output types
        answer = response.get("answer", response)

        self.history.append(HumanMessage(content=human_message))                
        self.history.append(AIMessage(content=answer))
        return answer

    def reset(self) -> None:
        self.history = []
        self.history.append(SystemMessage(content=self.system_prompt))

    def get_history(self) -> List[BaseMessage]:
        return self.history
    
    def retrieve(self, question: str, k:int = 4) -> List[Document]:
        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=k)
            return retrieved_docs
        except Exception as e:
            print(f"⚠️ Retrieval Error: {e}")
            return []  # ✅ CHANGED — added safety fallback

    def add_pdf_to_context(self, filePath: Path) -> List[Document]:
        return self.vector_store.add_document(filePath)

if __name__ == '__main__':
    print(ChatOllama.list_models())
