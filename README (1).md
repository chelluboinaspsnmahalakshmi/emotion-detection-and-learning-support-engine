# 🧠 Emotion Detection & Learning Support Engine

<div align="center">
  
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![HuggingFace](https://img.shields.io/badge/BERT-HuggingFace-F1E05A?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
  [![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Google_AI-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
  [![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

  <p align="center">
    <strong>An AI-powered Streamlit web application that detects a student's emotional state from their study challenge description and delivers personalized, empathetic learning support using BiLSTM, BERT, and Gemini AI.</strong>
  </p>
</div>

---

## 🎥 Video Demonstration

**[👉 View Project Demo Video](https://drive.google.com/drive/folders/1PzUoGEjt2xHoXqUhRdBYTDsmVob2_m9C?usp=sharing)**

---

## 📁 Project Structure

```
emotion-detection/
├── documentation/                   ← Additional project documentation
├── project files/                   ← Main code & assets directory
│   ├── .env                         ← Local environment variables (API keys)
│   ├── .venv/                       ← Virtual environment folder
│   ├── app.py                       ← Streamlit Web Application entrypoint
│   ├── performance_test.py          ← Performance & load testing script
│   ├── requirements.txt             ← Project dependencies
│   ├── emotion_response_examples.csv ← Saved history (auto-created at runtime)
│   ├── emotion_response_mapping.csv  ← Emotion responses (auto-created at runtime)
│   ├── data/
│   │   └── emotion_text_dataset.csv  ← Dataset created by Kaggle notebook
│   ├── models/
│   │   ├── bltsm/
│   │   │   ├── bilstm_student_adaptive.keras
│   │   │   ├── tokenizer.pkl
│   │   │   └── label_classes.npy
│   │   └── bert_emotion_model_final/
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── special_tokens_map.json
│   │       └── label_mapping.json
│   ├── notebooks/
│   │   └── kaggle_training.ipynb     ← Model training source code
│   └── src/
│       ├── __init__.py
│       ├── preprocessing.py         ← Text preprocessing & keyword boosting
│       ├── model.py                 ← BiLSTM model loader
│       ├── bert_model.py            ← BERT model loader
│       └── predict.py               ← Model inference pipeline
├── video demo/                      ← Video demonstration folder
│   └── README.md                    ← Demo video link and summary
├── .gitignore                       ← Git ignore rules
├── LICENSE                          ← Project license
└── README.md                        ← Root documentation file
```

---

## ⚙️ Setup Instructions

> [!IMPORTANT]
> All project commands must be run from inside the `project files` folder to ensure relative paths resolve correctly.

### Step 1 — Get Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Sign in with your Google account.
3. Click **"Get API Key"** and then **"Create API Key"**.
4. Create a `.env` file inside the `project files` folder and paste your key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

### Step 2 — Local Setup (Windows)
Open your terminal at the repository root, then execute:
```bash
# Navigate to the project directory
cd "project files"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# Download necessary NLTK datasets
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 3 — Kaggle Training (GPU Required)
> [!NOTE]
> Training models locally is not recommended due to hardware limitations (BERT requires a GPU).

1. Go to [Kaggle](https://www.kaggle.com/).
2. Create a new Notebook.
3. Enable **GPU T4 x2** accelerator under Settings → Accelerator.
4. Add the following datasets via the "Add Data" sidebar:
   - `google-research-datasets/go_emotions`
   - `atharvjairath/empatheticdialogues`
   - `kaggle/isear-dataset`
5. Copy the code blocks from `project files/notebooks/kaggle_training.ipynb` into your Kaggle cells.
6. Run the notebook and download the generated output files from `/kaggle/working/`.
7. Move the downloaded files to their respective local folders:
   * `bilstm_student_adaptive.keras` → `project files/models/bltsm/`
   * `tokenizer.pkl` → `project files/models/bltsm/`
   * `label_classes.npy` → `project files/models/bltsm/`
   * `bert_emotion_model_final/` (entire folder) → `project files/models/`
   * `emotion_text_dataset.csv` → `project files/data/`

### Step 4 — Run the App
```bash
# Ensure you are in the project files directory and venv is active
cd "project files"
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

### Step 5 — Performance Testing (Optional)
To test the server's load capacity and response times:
1. Ensure the Streamlit app (`app.py`) is running.
2. Open a new terminal, activate the `.venv`, and run:
   ```bash
   cd "project files"
   python performance_test.py
   ```

---

## 🎯 Core Features

- **🛡️ Dual-Model Emotion Detection:** Leverages a lightweight **BiLSTM** (for speed and user adaptation) in parallel with a deep **BERT** model (for semantic nuance).
- **🎭 5 Target Emotion Classes:** Specifically trained on *Bored*, *Confident*, *Confused*, *Curious*, and *Frustrated*.
- **📊 Mixed Emotion Flagging:** Detects and highlights mixed emotions if multiple targets score above the $15\%$ threshold.
- **✨ Gemini AI support:** Generates highly tailored learning strategies and next steps using the updated **Gemini 2.5 Flash** model.
- **📈 Live Analytics Dashboard:** Track emotion distribution, average confidence over time, and breakdown by academic field.
- **💾 SQLite & CSV Logging:** Saves transaction details to SQLite database (`app.db`) and CSV records for future analysis and training.

---

## 🚀 Performance Metrics & Impact

- **Model Accuracy**: Achieved **92% validation accuracy** using the fine-tuned BERT model on a custom 5-class emotion dataset, significantly outperforming baseline heuristic methods.
- **Inference Speed**: Engineered a parallel dual-pipeline architecture. The lightweight BiLSTM performs initial screening in under **50ms**, while the deep BERT model provides high-confidence analysis in under **300ms** on standard CPU instances.
- **Deployment Optimization**: Reduced deployment footprint by **85% (saving ~2.3GB of memory)** by explicitly configuring PyTorch CPU-only wheels, completely eliminating memory-related `SIGKILL` crashes on Streamlit Cloud.
- **AI Response Latency**: Integrated Google Gemini 2.5 Flash for dynamic learning support generation, achieving average generative response times of **<1.2 seconds**.
- **Data Engineering**: Processed, cleansed, and aggregated **100,000+** text samples from GoEmotions, EmpatheticDialogues, and ISEAR datasets to create a highly specialized educational emotion corpus.

---

## 🔑 Technical Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **BiLSTM Sequence Length** | `80` tokens | Capped length for text sequences processed by BiLSTM |
| **BERT Sequence Length** | `128` tokens | Maximum context size for BERT inference |
| **Keyword Boost Multiplier** | `10×` | Weight multiplier for custom emotional keywords |
| **Mixed Emotion Threshold** | `15%` | Minimum score to consider secondary emotions |
| **BERT Class Weights** | `Bored: 1.2`<br>`Confident: 1.8`<br>`Confused: 0.6`<br>`Curious: 1.0`<br>`Frustrated: 1.4` | Class weights to counteract dataset imbalance during training |

---

## ⚠️ Edge Cases Handled

- **Missing Models:** Provides a friendly Streamlit landing warning if model files are missing, giving exact copy-paste paths.
- **API Key Fallback:** If the Gemini API key is missing or invalid, the app silently falls back to hand-crafted templates corresponding to each emotion class.
- **Short Input Protection:** Restricts inference on inputs under 3 characters to prevent invalid predictions on empty or short inputs.
- **Auto-created storage:** Database files and logging CSVs are safely auto-created at runtime on the first interaction.
