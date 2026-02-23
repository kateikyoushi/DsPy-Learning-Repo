# Cebu Pacific AI Support Chatbot

✈️ **Cebu Pacific AI Support Chatbot** - Powered by DSPy-Optimized AI (64% Quality)

A single-file Streamlit application showcasing an optimized customer support agent for Cebu Pacific Airlines, built using DSPy's MIPROv2 optimization algorithm. The agent achieves a **31% improvement** in response quality (from 49% to 64%).

## Features

- 💬 Real-time chatbot with optimized AI agent
- 📊 Performance analytics and business impact calculator
- 🔬 MLflow experiment tracking and logs viewer
- 📄 Knowledge base review of scraped information
- ℹ️ Comprehensive documentation and use cases
- 🎨 Clean UI using default Streamlit styling

## Technology Stack

- **Framework**: Streamlit
- **AI Optimization**: DSPy with MIPROv2
- **LLM**: Google Gemini 3 Flash Preview
- **Experiment Tracking**: MLflow
- **Visualizations**: Plotly
- **Data Processing**: Pandas

## Local Development

### Prerequisites

- Python 3.11+
- Google Gemini API key

### Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   For local development, create a `.env` file in the root directory:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   MLFLOW_TRACKING_URI=http://localhost:8080  # Optional, for local MLflow
   ```

   For Streamlit Cloud deployment, set these in the app's secrets instead.

5. Run the application:
   ```bash
   streamlit run ceb_pac_stream.py
   ```

### Optional: Run MLflow Server

For full MLflow integration, start a local MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

## Deployment to Streamlit Cloud

1. **Rename the main file**: Rename `ceb_pac_stream.py` to `app.py` (Streamlit Cloud expects `app.py` by default).

2. **Set up secrets**: In your Streamlit Cloud dashboard, add the following secrets:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `MLFLOW_TRACKING_URI`: (Optional) Your MLflow tracking URI if using remote MLflow

3. **Push to GitHub**: Ensure the following files are in your GitHub repository:
   - `app.py` (renamed from `ceb_pac_stream.py`)
   - `requirements.txt`
   - `README.md`
   - `cebu_pacific_optimized_agent_flash.json`
   - `optimization_results.json` (optional, for cached results)
   - `cebu_pacific_helpcenter.md`

4. **Deploy**: Connect your GitHub repo to Streamlit Cloud and deploy. The app will automatically install dependencies from `requirements.txt`.

## Files Required for Deployment

Upload these files to your GitHub repository:

- `app.py` (the main Streamlit app, renamed from `ceb_pac_stream.py`)
- `requirements.txt` (generated dependencies list)
- `README.md` (this file)
- `cebu_pacific_optimized_agent_flash.json` (optimized agent parameters)
- `optimization_results.json` (optimization metrics, optional but recommended)
- `cebu_pacific_helpcenter.md` (scraped knowledge base content)

## Configuration

The app uses the following configuration (defined in the code):

- Agent file: `cebu_pacific_optimized_agent_flash.json`
- Results file: `optimization_results.json`
- Help center file: `cebu_pacific_helpcenter.md`
- MLflow URI: Defaults to `http://localhost:8080` (set via environment variable)

## Usage

1. **Chat Tab**: Interact with the AI agent, ask questions about Cebu Pacific services.
2. **Analytics Tab**: View performance metrics and business impact calculations.
3. **MLflow Logs Tab**: Explore optimization experiments (requires MLflow server).
4. **Knowledge Base Tab**: Review the scraped help center information.
5. **About Tab**: Learn more about the project and technology.

## Performance

- **Quality Score**: 64% (up from 49% baseline)
- **Improvement**: 31%
- **Response Time**: ~5.4 seconds (down from 6.2 seconds)
- **Annual Cost Savings**: $821,250 (estimated)

## License

MIT License

## Version

1.0.0 (February 2026)