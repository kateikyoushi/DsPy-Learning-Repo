# app.py - Ilonggo Dictionary Translator Streamlit Chatbot
# Run with: streamlit run app.py

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration
st.set_page_config(
    page_title="🇵🇭 Ilonggo Translator",
    page_icon="🇵🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "translator_loaded" not in st.session_state:
    st.session_state.translator_loaded = False

# Cache the translator setup
@st.cache_resource
def load_translator():
    """Load the translation system (cached for performance)"""
    try:
        # Configure Gemini LLM (FIXED MODEL NAME)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # ✅ FIXED - removed "gemini/" prefix
            google_api_key=os.getenv('GEMINI_API_KEY'),
            temperature=0.3,
            convert_system_message_to_human=True  # Add this for compatibility
        )
        
        # Load local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load FAISS vectorstore
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Custom prompt template
        template = """You are an Ilonggo-English dictionary assistant. Use the dictionary entries below to help translate.

Dictionary Context:
{context}

User Question: {question}

Instructions:
- If the word is in the dictionary, provide the definition/translation
- If it's English to Ilonggo, search for the English word
- If it's Ilonggo to English, search for the Ilonggo word
- If not found, say "I couldn't find that word in the dictionary"
- Be helpful, friendly, and concise
- Provide pronunciation help if available in the dictionary

Translation:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Format docs function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build translation chain
        translator_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return translator_chain, retriever, vectorstore, llm
    
    except FileNotFoundError:
        st.error("❌ FAISS index not found!")
        st.info("Please run the Jupyter notebook (Cells 1-7) first to create the vector database.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading translator: {e}")
        st.stop()

# Main app
def main():
    # Header
    st.title("🇵🇭 Ilonggo Dictionary Translator")
    st.markdown("*Translate between Ilonggo and English using RAG + Local Embeddings*")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This translator uses RAG (Retrieval-Augmented Generation) "
            "with your PDF dictionaries and FAISS vector search."
        )
        
        # Load translator
        with st.spinner("Loading translator..."):
            translator_chain, retriever, vectorstore, llm = load_translator()
            st.session_state.translator_loaded = True
        
        st.success("✅ Translator loaded!")
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Statistics")
        st.metric("Dictionary Entries", f"{vectorstore.index.ntotal:,}")
        st.metric("Chat Messages", len(st.session_state.messages))
        st.metric("Model", "Gemini 2.5 Flash")
        st.metric("Embeddings", "Local (Unlimited)")
        
        # Instructions
        st.markdown("---")
        st.subheader("💡 How to Use")
        st.markdown("""
        1. Type your question in the chat
        2. Get instant translation
        3. View dictionary sources (optional)
        4. Clear chat to start fresh
        """)
        
        # Example queries
        st.markdown("---")
        st.subheader("📝 Example Questions")
        example_queries = [
            "What does 'mahal' mean?",
            "How do you say 'hello' in Ilonggo?",
            "Translate 'kumusta' to English",
            "What is the Ilonggo word for beautiful?"
        ]
        
        for query in example_queries:
            if st.button(query, key=query, use_container_width=True):
                # Add to chat
                st.session_state.messages.append({"role": "user", "content": query})
                # Trigger rerun to process
                st.rerun()
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 View {len(message['sources'])} dictionary sources"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.text(f"Source {i}:\n{doc.page_content[:300]}...")
                        if i < len(message["sources"]):
                            st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask for a translation..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Translating..."):
                try:
                    # Get translation
                    response = translator_chain.invoke(prompt)
                    
                    # Get source documents
                    sources = retriever.invoke(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add sources expander
                    with st.expander(f"📚 View {len(sources)} dictionary sources"):
                        for i, doc in enumerate(sources, 1):
                            st.text(f"Source {i}:\n{doc.page_content[:300]}...")
                            if i < len(sources):
                                st.markdown("---")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🇵🇭 Built with Streamlit • Powered by Gemini 2.5 Flash • Local Embeddings
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()
