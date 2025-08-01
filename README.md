# Fake News Generator & Detector

A simple yet powerful **Fake News Generator and Detector** web application powered by state-of-the-art Natural Language Processing (NLP) models.

This project combines two AI components:

- **Fake News Generator:** Uses GPT-2, a generative language model, to create synthetic news articles from any headline or prompt.
- **Fake News Detector:** Uses a multilingual BERT classifier to analyze news text and estimate whether it is likely *Real* or *Fake* news.

## 🚀 Features

- Generate fake news articles by simply typing a headline or prompt.
- Detect whether a news article is likely real or fake based on model confidence scores.
- Choose writing styles like "Neutral", "Sensational", or "Satirical" to control the tone of generated news.
- Supports multi-language inputs (English, Spanish, French) for both generation and detection.
- Interactive visualization showing confidence levels in the detection results.
- Clean and responsive browser interface built with [Gradio](https://gradio.app/).
- Runs entirely on CPU, no GPU required.

## 🔧 How to Use

### Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install torch transformers gradio plotly
```

### Running Locally

1. Clone or download this repository.
2. Navigate to the project folder.
3. Run the app:

```bash
python app.py
```

4. Open your browser and visit the URL printed in the terminal (usually http://127.0.0.1:7860).
5. Use the interface to generate fake news or detect authenticity of any news text.

### Running on Google Colab

- Copy the code into a notebook cell.
- Run the cell with `!pip install` commands to install requirements.
- Launch the Gradio interface with `demo.launch(share=True)` to get a publicly shareable link.

## 🧠 Models Used

- **News Generator:** GPT-2 from Hugging Face Transformers for text generation.
- **News Detector:** Multilingual BERT (bert-base-multilingual-uncased) for fake news classification.

> **Note:** The detection model is a base pre-trained model and hasn’t been fine-tuned on labeled fake news data. For reliable results, fine-tuning on a domain-specific fake news dataset is recommended.

## 🎨 UI Overview

- Side-by-side panels for news generation and detection.
- Dropdown menus for language and writing style selection.
- Sliders for tuning generation length and creativity.
- Confidence pie chart visualizes detector prediction probabilities.
- Sample prompts and articles provided for quick testing.

## ⚠️ Disclaimer

- This project is for **research, educational, and experimental purposes only**.
- The generator produces synthetic content that is **not factual**.
- The detector model is not yet fine-tuned, so results should not be relied upon for real-world decisions.
- Always verify information through trusted sources.

## 📂 Project Structure

```
fake-news-generator-detector/
│
├── app.py                  # Main Gradio app code
├── README.md               # This readme file
└── requirements.txt        # List of Python dependencies (optional)
```

## ✨ Future Improvements

- Fine-tune the detector model on labeled fake news datasets for better accuracy.
- Add support for more languages and larger generative models.
- Provide user feedback loops and explainability insights.
- Deploy the app on cloud platforms for easy sharing.

Hope this helps you clearly document your project on GitHub! Let me know if you want me to help prepare a `requirements.txt` or deployment instructions.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/86505336/b297030f-b254-4d10-9897-31694f6694b6/app.py
