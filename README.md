# AGB_Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about legal terms and conditions (AGBs) from PDF documents. It uses language models (e.g., GPT-3.5-turbo or Ollama models) and embeddings to retrieve relevant information and provide accurate responses.

## Files and Structure

- **`agb_chatbot.py`**: Core script to set up the chatbot. Loads PDFs from the "B2B" directory, processes them into a vector store, and defines a question-answering chain.
- **`evaluation_giskard.py`**: Evaluates the chatbot using the Giskard library and generates an HTML report (`test2.html`).
- **`evaluations_ragas.py`**: Performs detailed evaluation using RAGAS metrics (e.g., faithfulness, relevancy, correctness).
- **`evaluation_speed.py`**: Measures the speed of retrieval and answer generation.
- **`B2B/`**: Directory containing PDF files with terms and conditions (AGBs) that form the knowledge base.

## Setup Instructions

1. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

   - Ensure you have Ollama installed locally if using Ollama models (mistral, llama3.1, etc.).
   
   - For OpenAI models, create a .env file with your OPENAI_API_KEY.

2. **Prepare PDFs**:

   - Place AGB PDF files in the B2B directory.

3. **Run the Chatbot**:

    ```
    python agb_chatbot.py
    ```
   - Edit the MODEL variable in agb_chatbot.py to switch between GPT or Ollama models.

4. **Evaluate the Chatbot**:

   - Run evaluation_giskard.py to generate a test set and evaluate:

      ```
      python evaluation_giskard.py
      ```

   - Run evaluations_ragas.py for RAGAS metrics (requires prior Giskard evaluation).

      ```
      python evaluation_ragas.py
      ```

   - Run evaluation_speed.py to measure performance.

      ```
      python evaluation_speed.py
      ```
      
5. **Use as Streamlit Web Interface (Streamlit App)**
   
      ```
      streamlit run app.py
      ```


## Usage Example

Ask the chatbot a question like:

- "Can I cancel my contract with Sunrise Mobile before the end of the minimum contract period?" The response will be based solely on the provided PDF context.
