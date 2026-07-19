import streamlit as st
import heapq
import numpy as np
import ollama
import time
import re
from typing import List, Dict, Tuple, Any, Iterator, Union
from datetime import datetime
from memory_monitor import render_memory_bar
from load_documents import load_all_from_docs_folder, load_uploaded_file, list_loaded_documents

# ── Default built-in content (used if no docs folder documents are found) ──────
# This is a compact summary of NIST CSF 2.0 key concepts.
# It ensures the system is never empty even before you place a PDF in docs/.
DEFAULT_CYBERSECURITY_INFO = """The NIST Cybersecurity Framework (CSF) 2.0 provides guidance to industry, government agencies, and other organizations to manage cybersecurity risks.
The CSF 2.0 is organized around six core Functions: Govern, Identify, Protect, Detect, Respond, and Recover.
The Govern (GV) Function establishes and monitors the organization's cybersecurity risk management strategy, expectations, and policy.
The Identify (ID) Function helps the organization understand its current cybersecurity risks to systems, people, assets, data, and capabilities.
The Protect (PR) Function uses safeguards to prevent or reduce cybersecurity risks including access control, data security, and training.
The Detect (DE) Function finds and analyzes possible cybersecurity attacks and compromises through continuous monitoring.
The Respond (RS) Function takes action regarding a detected cybersecurity incident including incident management and communication.
The Recover (RC) Function restores assets and operations that were impacted by a cybersecurity incident including recovery planning and improvements.
The CSF 2.0 introduced the Govern function as a new addition over CSF 1.1, reflecting the importance of cybersecurity governance at the executive and board level.
Organizational Context (GV.OC) ensures that the organizational mission is understood and that cybersecurity risk management supports and shapes the organization's objectives.
Risk Management Strategy (GV.RM) ensures that the organization's priorities, constraints, risk tolerance, and assumptions are established, communicated, and used to support cybersecurity risk decisions.
Asset Management (ID.AM) involves identifying and managing assets (data, hardware, software, systems, and facilities) that enable the organization to achieve business purposes.
Risk Assessment (ID.RA) identifies and prioritizes risks to the organization's operations, assets, and individuals based on threat and vulnerability information.
Identity Management and Access Control (PR.AA) ensures that access to physical and logical assets is limited to authorized users, processes, and devices.
Awareness and Training (PR.AT) ensures that personnel are provided cybersecurity awareness education and trained to perform their cybersecurity-related duties.
Data Security (PR.DS) ensures that data-at-rest, in-transit, and in-use are managed in accordance with the organization's risk strategy to protect confidentiality, integrity, and availability.
Platform Security (PR.PS) manages the hardware, software, and services of physical and virtual platforms according to the organization's risk strategy to protect confidentiality, integrity, and availability.
Technology Infrastructure Resilience (PR.IR) ensures that security architectures are managed with the organization's risk strategy to protect asset confidentiality, integrity, and availability.
Adverse Event Analysis (DE.AE) ensures that anomalies, indicators of compromise, and other potentially adverse events are analyzed to characterize the events and detect cybersecurity incidents.
Incident Management (RS.MA) covers the response to detected cybersecurity incidents including the execution of the response plan and activities to minimize the impact.
Incident Analysis (RS.AN) ensures that investigations are conducted to ensure effective response and support forensics and recovery activities.
Incident Response Reporting and Communication (RS.CO) covers response activities that are coordinated with internal and external stakeholders as required by laws, regulations, or policies.
Incident Mitigation (RS.MI) ensures that activities are performed to prevent expansion of an event and mitigate its effects.
Incident Recovery Plan Execution (RC.RP) ensures that restoration activities are performed to ensure operational availability of systems and services affected by cybersecurity incidents.
Incident Recovery Communication (RC.CO) ensures that restoration activities are coordinated with internal and external parties such as coordinating centers, Internet Service Providers, vendors, and other CSIRTs.
The CSF Core provides a set of cybersecurity outcomes organized as Functions, Categories, and Subcategories that can be customized for any organization.
CSF Profiles represent the alignment of the CSF Core to an organization's requirements, risk tolerance, and resources to support prioritization of cybersecurity activities.
CSF Tiers describe the maturity of an organization's cybersecurity risk governance and management practices ranging from Tier 1 (Partial) to Tier 4 (Adaptive).
Supply chain risk management is addressed under the Govern function to ensure cybersecurity risks in the supply chain are identified, assessed, and managed."""


class ConvoRAG:
    def __init__(
        self,
        documents: List[str],
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        st.write(f"Initializing with {len(documents)} document chunks")
        st.write(
            f"Using embedding model: {embedding_model}, LLM model: {llm_model}"
        )

        with st.spinner("Embedding documents... (this may take a minute for large documents)"):
            self.document_embeddings = [self.embed_text(doc) for doc in documents]

        self.conversation_history = []

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            embedding_array = np.array(embedding)
            st.write(
                f"Embedded text '{text[:50]}...' → {embedding_array.shape} dimensions"
            )
            return embedding_array
        except Exception as e:
            st.error(f"Error in embedding: {str(e)}")
            return np.zeros(768)

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return similarity
        except Exception as e:
            st.error(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0

    def topk(self, arr: List[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        if not arr:
            return []

        k = min(k, len(arr))

        try:
            topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
            top_scores = [arr[i] for i in topk_indices]
            st.write(
                f"Top {k} similarity scores: {[f'{score:.4f}' for score in top_scores]}"
            )
            return topk_indices
        except Exception as e:
            st.error(f"Error in topk calculation: {str(e)}")
            return list(range(min(k, len(arr))))

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity of embeddings."""
        if not self.documents or not self.document_embeddings:
            return "No documents available.", 0.0

        try:
            st.write(f"Searching for: '{query}'")
            query_embedding = self.embed_text(query)

            similarities = []
            for doc_embedding in self.document_embeddings:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)

            if not similarities:
                return "No similarities found.", 0.0

            topk_indices = self.topk(similarities, top_k)

            if not topk_indices:
                return "No relevant documents found.", 0.0

            result = "\n".join([self.documents[i] for i in topk_indices])
            return result, similarities[topk_indices[0]] if topk_indices else 0.0
        except Exception as e:
            st.error(f"Error in document search: {str(e)}")
            return "Error occurred while searching documents.", 0.0

    def generate_answer(
        self, system_prompt: str, user_prompt: str, is_stream: bool = True
    ) -> Any:
        """Generate a response using the provided system and user prompts."""
        try:
            st.write(f"Generating response with model: {self.llm_model}")

            generation_start = time.time()
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=is_stream,
            )

            if not is_stream:
                generation_time = time.time() - generation_start
                st.write(
                    f"Response generated in {generation_time:.2f} seconds"
                )

            return response if is_stream else (response["message"]["content"])
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if is_stream:
                return iter([{"message": {"content": error_msg}}])
            else:
                return error_msg

    def detect_query_type(self, query: str, relevance_threshold: float = 0.75) -> str:
        """Use LLM to classify the query type into cybersecurity-related, compliment, complaint, chitchat, or off-topic."""
        st.write(f"Detecting query type for: '{query}'")

        system_prompt = """
        You are an expert at classifying cybersecurity and information security queries. Your task is to categorize each query into EXACTLY ONE of these categories:
        
        1. "cybersecurity-related" - Questions about cybersecurity standards, frameworks, controls, policies, risk management, compliance, threats, vulnerabilities, security functions, governance, etc.
        Examples:
        - "What are the six functions of NIST CSF 2.0?"
        - "What does the Govern function cover?"
        - "How do I implement access control?"
        - "What is the Protect function?"
        - "What is supply chain risk management?"
        - "How does the Detect function work?"
        - "What are CSF Tiers?"
        - "What is a CSF Profile?"
        - "How do I recover from a cybersecurity incident?"
        - "What subcategories does Identify cover?"
        
        2. "compliment" - Positive feedback or appreciation
        Examples:
        - "Thank you for the information"
        - "That's very helpful"
        - "Great explanation"
        - "You're doing a great job"
        - "You're providing relevant information"
        
        3. "complaint" - Negative feedback or dissatisfaction
        Examples:
        - "That's not what I asked"
        - "I'm not happy with your answer"
        - "Your answer is wrong"
        - "You are responding slow"
        - "That's not the answer I expected"
        
        4. "chitchat" - General conversation not directly related to cybersecurity
        Examples:
        - "How are you today?"
        - "What's your name?"
        - "Tell me a joke"
        - "How was your day?"
        - "Hey, nice to meet you!"
        
        5. "off-topic" - Questions completely unrelated to cybersecurity or information security
        Examples:
        - "How do I fix my car?"
        - "What's the capital of France?"
        - "Can you write me a poem?"
        - "What is the hotel check-in time?"
        - "Tell me about cooking recipes"
        
        IMPORTANT RULES:
        - Follow-up questions about cybersecurity topics should be classified as "cybersecurity-related" even if they are brief
        - Questions starting with "Is it...", "Does it...", "How much...", "What is..." often refer to previously mentioned cybersecurity topics
        - Treat ambiguous queries that could reasonably be about cybersecurity as "cybersecurity-related"
        - WHEN IN DOUBT, classify as "cybersecurity-related"
        
        Return ONLY the category name, with no explanation.
        """

        user_prompt = f'Classify this query: "{query}"'

        try:
            initial_classification = (
                self.generate_answer(system_prompt, user_prompt, is_stream=False)
                .lower()
                .strip()
            )

            if "cybersecurity" in initial_classification:
                classification = "cybersecurity-related"
            elif "compliment" in initial_classification:
                classification = "compliment"
            elif "complaint" in initial_classification:
                classification = "complaint"
            elif "chitchat" in initial_classification:
                classification = "chitchat"
            else:
                classification = "off-topic"

            st.write(f"Initial classification: '{classification}'")

            if classification == "cybersecurity-related":
                return classification

            # For potentially misclassified queries, do semantic search to verify
            if classification in ["chitchat", "off-topic"]:
                relevant_docs, top_similarity = self.search(query, top_k=1)

                st.write(
                    f"Query: '{query}' | Initial classification: {classification} | Top similarity: {top_similarity:.4f}"
                )

                if top_similarity >= relevance_threshold:
                    st.write(
                        f"Reclassifying to cybersecurity-related due to high document similarity: {top_similarity:.4f}"
                    )
                    return "cybersecurity-related"

            return classification

        except Exception as e:
            st.error(f"Error in query classification: {str(e)}")
            return "cybersecurity-related"

    def contextualize_query(self, current_query: str) -> str:
        """Enhance the current query with context from conversation history."""
        if not self.conversation_history:
            return current_query

        st.write(f"Contextualizing query: '{current_query}'")

        if len(current_query.split()) > 10:
            st.write(f"Query is detailed enough, no contextualization needed")
            return current_query

        history_context = (
            "The conversation history is ordered from oldest to most recent:\n\n"
        )
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx + 1}:\nUser: {q}\nAssistant: {a}\n\n"

        system_prompt = """You are a query reformulation system for a cybersecurity standards assistant. Your job is to take a user's query and reformulate it to be self-sufficient by incorporating relevant context from the conversation history, while preserving the original intent.

RULES:
1. Create a single, concise sentence that captures both the current query and relevant context
2. Never add your own opinions, reasoning, explanations, or commentary
3. Preserve all details from the user's current query
4. Be sensitive to topic changes - when a topic changes, do not carry over unrelated context
5. For ambiguous references, prioritize the most relevant (not necessarily the most recent) conversation
6. Never answer the question - only reformulate it
7. If the query is already self-sufficient, make minimal changes or return it as is
8. Use natural language as if the user had included all relevant context themselves
9. If the user's intent is a question, always ensure the reformulated text is also a question

EXAMPLES:

Example 1 - Basic context addition:
User: "What does the Protect function cover?"
Reformulated query: "What areas and subcategories does the NIST CSF 2.0 Protect (PR) function cover?"

Example 2 - Topic change recognition:
User: "What does the Protect function cover?"
Assistant: [Reply on Protect function...]
User: "What about Detect?"
Reformulated query: "What does the NIST CSF 2.0 Detect (DE) function cover?"
DO NOT: "What does the Protect function and Detect function cover?" [INCORRECT - forces old topic into new question]

Example 3 - Pronoun resolution:
User: "What is the Govern function?"
Assistant: [Reply on Govern function...]
User: "Is it new in CSF 2.0?"
Reformulated query: "Is the Govern function new in NIST CSF 2.0?"

Example 4 - Preserve question format:
User: "Tell me about CSF Tiers."
Assistant: [Reply on CSF Tiers...]
User: "How many are there?"
Reformulated query: "How many Tiers are defined in NIST CSF 2.0?"

DO NOT add commentary or explanations.
DO NOT exceed one sentence unless absolutely necessary.
Always maintain the question format when the user's intent is a question."""

        user_prompt = f"""
Conversation history:
{history_context}

Current query: "{current_query}"

Reformulated query:
        """
        try:
            reformulated_query = self.generate_answer(
                system_prompt, user_prompt, is_stream=False
            ).strip()
            st.write(f"Reformulated query: '{reformulated_query}'")

            if ":" in reformulated_query:
                parts = reformulated_query.split(":", 1)
                if len(parts) > 1:
                    reformulated_query = parts[1].strip()

            reformulated_query = reformulated_query.strip("\"'")

            problematic_phrases = [
                "based on",
                "according to",
                "from our conversation",
                "as mentioned",
                "earlier you asked",
                "you asked about",
            ]

            if (
                len(reformulated_query.split()) > 25
                or any(
                    phrase in reformulated_query.lower()
                    for phrase in problematic_phrases
                )
                or ("." in reformulated_query and "?" not in reformulated_query)
            ):
                st.write(
                    f"Reformulation looks problematic, falling back to original query"
                )
                return current_query

            if reformulated_query.lower() == current_query.lower():
                st.write(f"Reformulation is identical to original query")
                return current_query

            return reformulated_query
        except Exception as e:
            st.error(f"Error in query contextualization: {str(e)}")
            return current_query

    def handle_non_cybersecurity_query(
        self, query_type: str, query: str
    ) -> Union[Iterator[Dict], str]:
        """Handle compliments, complaints, chitchat, and off-topic queries using LLM."""
        st.write(f"Handling {query_type} query: '{query}'")

        if query_type == "compliment":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has given you a compliment or thanked you.
            
            Respond with:
            1. A brief, gracious acknowledgment of their thanks
            2. An offer to help with any cybersecurity standards or framework questions they might have
            
            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User compliment: "{query}"'

        elif query_type == "complaint":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has expressed dissatisfaction or complained.
            
            Respond with:
            1. A brief, professional apology for any confusion
            2. An offer to help with specific cybersecurity standards questions
            
            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User complaint: "{query}"'

        elif query_type == "chitchat":
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has engaged in general conversation not related to cybersecurity.
            
            Respond with:
            1. A brief, friendly response
            2. A polite redirection to cybersecurity-related topics
            
            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'User message: "{query}"'

        else:  # off-topic
            system_prompt = """
            You are an AI assistant for a Cybersecurity Standards information system. A user has asked about a topic completely unrelated to cybersecurity.
            
            Respond with:
            1. A polite explanation that you are designed to assist with cybersecurity standards and frameworks such as NIST CSF 2.0
            2. An offer to help with cybersecurity-related questions
            
            Keep your response to 1-2 sentences maximum.
            """
            user_prompt = f'Off-topic question: "{query}"'

        return self.generate_answer(system_prompt, user_prompt)

    def rag(self, query: str) -> Union[Iterator[Dict], str]:
        """Main function to perform conversational document search and answer generation."""
        try:
            with st.expander(f"🔍 Debug info for query: '{query}'", expanded=False):
                st.write(f"Processing query: '{query}'")

                query_type = self.detect_query_type(query)
                st.write(f"Query type determined: '{query_type}'")

                if query_type != "cybersecurity-related":
                    st.write(f"Handling non-cybersecurity query of type: {query_type}")
                    response_stream = self.handle_non_cybersecurity_query(query_type, query)

                    if not isinstance(response_stream, Iterator):
                        self.conversation_history.append((query, response_stream))
                        return iter([{"message": {"content": response_stream}}])

                    return response_stream

                contextualized_query = self.contextualize_query(query)
                st.write(f"Contextualized query: '{contextualized_query}'")

                context, relevance_score = self.search(contextualized_query)
                st.write(f"Top relevance score: {relevance_score:.4f}")

                if relevance_score < 0.35 and context != "No documents available.":
                    st.write(
                        f"Low relevance score ({relevance_score:.4f}), using fallback response"
                    )
                    system_prompt = """
                    You are an AI assistant specializing in cybersecurity standards and frameworks. You are designed to answer questions about frameworks such as NIST CSF 2.0, ISO 27001, and other cybersecurity standards.
                    
                    The user has asked a question that does not match the available document content. Respond with:
                    
                    1. A polite explanation that you do not have enough information in your current documents to answer their specific question
                    2. An offer to help with other cybersecurity standards questions
                    3. A suggestion to check the original standard documents for more specific information
                    
                    Keep your response brief and professional.
                    """

                    user_prompt = f'User question without matching context: "{query}"'

                    return self.generate_answer(system_prompt, user_prompt)

                st.write("Generating final response with retrieved context")
                system_prompt = """You are an expert AI assistant specializing in cybersecurity standards and frameworks. You have access to a collection of detailed information from cybersecurity standards documents such as the NIST Cybersecurity Framework (CSF) 2.0 and other official cybersecurity standards.

                Your task is to help users by providing accurate, clear, and well-structured answers about cybersecurity standards, controls, functions, categories, and best practices.

                The information you have includes:

                Cybersecurity Framework Functions: Govern (GV), Identify (ID), Protect (PR), Detect (DE), Respond (RS), Recover (RC) — their purposes, categories, and subcategories.
                Risk Management: Organizational risk strategy, risk assessment, risk tolerance, and risk treatment.
                Governance: Cybersecurity policies, roles, responsibilities, supply chain risk management, and oversight.
                Identity and Access Control: Asset management, user authentication, privilege management, and access policies.
                Protective Controls: Data security, platform security, awareness training, and technology infrastructure resilience.
                Detection and Analysis: Continuous monitoring, anomaly detection, and adverse event analysis.
                Incident Response and Recovery: Incident management, communication, mitigation, and recovery planning.
                CSF Profiles and Tiers: Framework customization and maturity assessment.

                If a user's question cannot be answered based on the provided context, respond with:

                "I cannot answer this based on the available documents. Please refer to the official standard documentation or consult your cybersecurity administrator for more information."

                Respond in a clear, structured, and professional manner. Use bullet points or numbered lists when explaining complex topics. Always base your answers on the provided document context.
                """

                user_prompt = f'Based on the following context from cybersecurity standards documents, please answer the question.\nIf the answer cannot be derived from the context, say "I cannot answer this based on the available documents." \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n'

                return self.generate_answer(system_prompt, user_prompt)

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your question. Please try asking again or consult your cybersecurity administrator for assistance."
            st.error(f"Error in RAG process: {str(e)}")
            return iter([{"message": {"content": error_msg}}])


def chunk_text_with_overlap(
    text: str, chunk_size: int = 450, overlap_size: int = 100
) -> List[str]:
    """Split text into chunks with overlap. Optimized for dense cybersecurity standards documents."""
    if not text:
        return ["No text provided"]

    words = text.split()
    chunks = []

    words_per_chunk = chunk_size
    overlap_words = overlap_size

    if not words:
        return ["No text provided"]

    if len(words) <= words_per_chunk:
        return [text]

    st.write(
        f"Chunking text with {len(words)} words into chunks of ~{words_per_chunk} words with {overlap_words} word overlap"
    )

    i = 0
    while i < len(words):
        chunk_end = min(i + words_per_chunk, len(words))
        chunk = " ".join(words[i:chunk_end])
        chunks.append(chunk)

        i += words_per_chunk - overlap_words

        if i + words_per_chunk > len(words) and i < len(words):
            chunks.append(" ".join(words[i:]))
            break

    improved_chunks = []
    for chunk in chunks:
        if len(chunk.split()) > (words_per_chunk / 2):
            for marker in ["\n\n", "\n", ". ", "! ", "? "]:
                if marker in chunk:
                    last_marker_pos = chunk.rfind(marker)
                    if last_marker_pos > int(
                        len(chunk) - overlap_size
                    ):
                        improved_chunks.append(chunk[: last_marker_pos + len(marker)])
                        break
            else:
                improved_chunks.append(chunk)
        else:
            improved_chunks.append(chunk)

    st.write(
        f"Created {len(chunks)} initial chunks, {len(improved_chunks)} improved chunks"
    )

    return improved_chunks if improved_chunks else chunks


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = True
    if "loaded_doc_names" not in st.session_state:
        st.session_state.loaded_doc_names = []


def display_chat_messages():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Process user input from the chat interface."""
    prompt = st.chat_input("Ask a question about cybersecurity standards (e.g., NIST CSF 2.0):")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        return prompt
    return None


def handle_document_upload() -> str:
    """Handle document upload (PDF or TXT) and return extracted text."""
    uploaded_file = st.file_uploader(
        "Upload a cybersecurity standards document (PDF or TXT)",
        type=["txt", "pdf"],
        disabled=st.session_state.document_uploaded,
        help="You can upload NIST CSF, ISO 27001, or any other cybersecurity standard document."
    )
    if uploaded_file is not None:
        return process_uploaded_document(uploaded_file)
    return None


def process_uploaded_document(uploaded_file) -> str:
    """Process the uploaded document (PDF or TXT) and extract its text."""
    try:
        document_text = load_uploaded_file(uploaded_file)
        char_count = len(document_text)
        word_count = len(document_text.split())
        st.write(f"Processed '{uploaded_file.name}': {char_count:,} characters, {word_count:,} words")
        return document_text
    except ImportError as e:
        st.error(str(e))
        return ""
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return ""


def stream_parser(stream):
    """Parse the streaming response from Ollama."""
    full_response = ""
    stream_start = time.time()
    chunk_count = 0

    for chunk in stream:
        chunk_count += 1
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            full_response += content
            yield content

    stream_duration = time.time() - stream_start
    if st.session_state.debug_mode:
        st.write(
            f"Response streaming completed in {stream_duration:.2f} seconds ({chunk_count} chunks)"
        )

    return full_response


def rag(query: str):
    """Perform RAG on the query using the initialized ConvoRAG system."""
    if st.session_state.rag_system:
        return st.session_state.rag_system.rag(query)
    else:
        return iter(
            [
                {
                    "message": {
                        "content": "Documents not processed yet. Please wait for the system to initialize or upload a document."
                    }
                }
            ]
        )


def main():
    """Main function to run the Streamlit app."""
    st.title("🔒 Cybersecurity Standards Assistant")
    st.caption("Powered by NIST CSF 2.0 and other cybersecurity standards")

    initialize_session_state()

    with st.sidebar:
        st.title("⚙️ Configuration")
        model = st.selectbox(
            "Select LLM Model",
            ["llama3.2", "qwen2.5:1.5b", "gemma2:2b", "phi4-mini", "gemma4:e2b"],
            index=0,
        )
        embed_model = st.selectbox(
            "Select Embedding Model", ["nomic-embed-text"], index=0
        )

        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=True)
        if st.session_state.debug_mode:
            st.info(
                "Debug mode is enabled. You will see detailed information about the system's operations."
            )

        # Show which documents are loaded
        if st.session_state.document_uploaded:
            st.markdown("---")
            st.subheader("📄 Loaded Standards")
            if st.session_state.loaded_doc_names:
                for doc_name in st.session_state.loaded_doc_names:
                    st.markdown(f"• {doc_name}")
            else:
                st.markdown("• Built-in NIST CSF 2.0 summary")
            st.markdown("---")
            st.subheader("🔐 About This System")
            st.markdown("""
            This assistant answers questions about:
            * **NIST CSF 2.0** — 6 Core Functions
            * **Govern** — Strategy & Governance
            * **Identify** — Asset & Risk Management
            * **Protect** — Safeguards & Controls
            * **Detect** — Monitoring & Analysis
            * **Respond** — Incident Response
            * **Recover** — Recovery Planning

            *Ask any question about cybersecurity standards and frameworks.*
            """)

    # Keep the RAG engine's model in sync with the sidebar selection
    if st.session_state.rag_system is not None:
        st.session_state.rag_system.llm_model = model

    # Render live memory usage bar
    render_memory_bar(current_model=model)

    if not st.session_state.document_uploaded:
        # ── Try loading from docs/ folder first ────────────────────────────
        docs_folder_text = load_all_from_docs_folder()
        doc_names = list_loaded_documents()

        if docs_folder_text.strip():
            # Documents found in docs/ folder — auto load them
            document_text = docs_folder_text
            st.session_state.loaded_doc_names = doc_names
            source_label = f"Auto-loaded {len(doc_names)} document(s) from docs/ folder: {', '.join(doc_names)}"
        else:
            # No docs in folder — fall back to built-in NIST CSF summary
            document_text = DEFAULT_CYBERSECURITY_INFO
            st.session_state.loaded_doc_names = []
            source_label = "Using built-in NIST CSF 2.0 summary (place your PDF in RAGStream/docs/ for full content)"

        with st.spinner(f"Processing cybersecurity documents..."):
            start_time = time.time()
            st.write(f"Document source: {source_label}")
            st.write(f"Starting document processing")

            # ── Optional: also allow additional upload on top of docs folder ──
            uploaded_text = handle_document_upload()
            if uploaded_text:
                document_text = document_text + "\n\n" + uploaded_text
                st.write("Merged uploaded document with existing content.")

            documents = chunk_text_with_overlap(
                text=document_text, chunk_size=450, overlap_size=100
            )
            st.session_state.documents = documents
            st.session_state.document_text = document_text

            st.write(
                f"Initializing RAG system with {len(documents)} document chunks"
            )
            st.session_state.rag_system = ConvoRAG(
                documents, embedding_model=embed_model, llm_model=model
            )
            st.session_state.document_uploaded = True

            welcome_msg = "👋 Welcome to the **Cybersecurity Standards Assistant**! I am here to help you with questions about cybersecurity frameworks and standards such as **NIST CSF 2.0**. You can ask me about the six core functions (Govern, Identify, Protect, Detect, Respond, Recover), risk management, controls, profiles, tiers, and much more. How can I assist you today?"
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )
            elapsed = time.time() - start_time
            st.success(f"Cybersecurity documents processed successfully! ({elapsed:.1f}s)")
            st.rerun()
    else:
        if st.sidebar.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            welcome_msg = "👋 Welcome to the **Cybersecurity Standards Assistant**! I am here to help you with questions about cybersecurity frameworks and standards such as **NIST CSF 2.0**. How can I assist you today?"
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )
            st.rerun()

        if st.sidebar.button("Restart & Reload Documents"):
            st.session_state.clear()
            st.rerun()

        display_chat_messages()
        user_input = handle_user_input()

        if user_input:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    message_placeholder.markdown("⏳ Analyzing your question...")

                    llm_stream = rag(user_input)

                    # Add the placeholder for the partial message immediately
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    stop_button_placeholder = st.empty()
                    with stop_button_placeholder:
                        # If the user clicks this button, Streamlit immediately reruns the app,
                        # stopping the current execution but keeping the partial message!
                        st.button("⏹ Stop Generation", key=f"stop_{len(st.session_state.messages)}")

                    collected_response = ""
                    for part in stream_parser(llm_stream):
                        full_response += part
                        collected_response += part
                        # Update session state continually so if stopped, partial message is kept
                        st.session_state.messages[-1]["content"] = collected_response
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)

                    message_placeholder.markdown(full_response)
                    # Clear the stop button when finished generating
                    stop_button_placeholder.empty()

                    st.session_state.rag_system.conversation_history.append(
                        (user_input, collected_response)
                    )

                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error. Please try asking again or consult your cybersecurity administrator for assistance."
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Cybersecurity Standards Assistant",
        page_icon="🔒",
        layout="wide"
    )
    main()
