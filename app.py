import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import plotly.graph_objs as go

device = torch.device("cpu")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

bert_model_name = "bert-base-multilingual-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=2).to(device)
bert_model.eval()

LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr"
}

GEN_STYLES = {
    "Neutral": 0.7,
    "Sensational": 1.1,
    "Satirical": 1.3,
}

LANG_DETECT_PROMPTS = {
    "en": "Analyze the following news article:",
    "es": "Analice el siguiente artículo de noticias:",
    "fr": "Analysez l'article de presse suivant:",
}

def generate_fake_news(prompt, max_length, temperature, style, language):
    if not prompt or len(prompt.strip()) < 5:
        return "⚠️ Please enter a longer, more descriptive news headline or prompt."
    style_temp = GEN_STYLES.get(style, 0.7)
    temp = max(min(float(temperature), 1.5), 0.1)
    final_temp = (style_temp + temp) / 2
    lang_hint = {
        "en": "",
        "es": " [Escribe en Español]",
        "fr": " [Écrire en Français]"
    }.get(language, "")
    full_prompt = prompt.strip() + lang_hint
    inputs = gpt2_tokenizer.encode(full_prompt, return_tensors="pt").to(device)
    outputs = gpt2_model.generate(
        inputs,
        max_length=int(max_length),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=final_temp,
        top_k=50,
        top_p=0.95,
        early_stopping=True
    )
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def detect_news(text, language):
    if not text or len(text.strip()) < 20:
        return "⚠️ Please paste a full news article or paragraph for best results.", None
    lang_code = LANGUAGES.get(language, "en")
    prompt_prefix = LANG_DETECT_PROMPTS.get(lang_code, LANG_DETECT_PROMPTS["en"])
    input_text = prompt_prefix + "\n" + text.strip()
    inputs = bert_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    fake_confidence = probabilities[0].item()
    real_confidence = probabilities[1].item()
    label = "🛑 Likely Fake" if fake_confidence > real_confidence else "✅ Likely Real"
    result_text = (
        f"### {label}\n"
        f"**Confidence (Fake):** {fake_confidence:.2f} &nbsp;|&nbsp; **Confidence (Real):** {real_confidence:.2f}"
    )
    confidence_data = {"Fake": fake_confidence, "Real": real_confidence}
    return result_text, confidence_data

def make_confidence_bar_chart(confidence_data):
    if confidence_data is None:
        return {}
    labels = list(confidence_data.keys())
    values = list(confidence_data.values())
    colors = ['#de425b', '#3fa44d']
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.5,
        sort=False,
        direction='clockwise',
        textinfo='label+percent',
        showlegend=False,
    )])
    fig.update_layout(margin=dict(t=0,b=0,l=0,r=0), height=250)
    return fig

CUSTOM_CSS = """
body { background-color: #f7fbfd; }
.gradio-container { font-family: 'Segoe UI', Arial, sans-serif; max-width:1100px !important; }
.tabitem {padding:1.5em;}
h1, h2, h3 { color: #243a5e; }
.gr-button {background: #0673be; color: #fff; border-radius:4px;}
.gr-button:hover {background: #45b2ff;}
textarea, input[type='text'] { background: #eef6f9; border-radius: 5px;}
.gr-slider { accent-color: #0673be;}
.sample-btn {background:#f1faff; color:#0673be;}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Fake News Generator & Detector") as demo:
    gr.Markdown(
        """
        # 📰 Fake News Generator & Detector<br>
        <span style='font-size:1.06em;color:#3a3a3a;'>
        Powered by GPT-2 & Multilingual BERT.<br>
        Create synthetic news for research—and check articles for authenticity in multiple languages.
        </span>
        <hr>
        """
    )
    lang_selector = gr.Dropdown(label="Select Language", choices=list(LANGUAGES.keys()), value="English")

    with gr.Row():
        with gr.Column(scale=1, min_width=350):
            gr.Markdown(
                """
                <h3>🛠 Generate Fake News</h3>
                <span style="color:#363636">
                Give a headline, and GPT-2 will write a full news story.<br>
                (Warning: For educational use only!)
                </span>
                """
            )
            input_text = gr.Textbox(
                label="News Headline / Prompt",
                placeholder="e.g. Alien spaceship lands in Times Square!",
                lines=2,
                info="Enter a creative, detailed event or headline."
            )
            max_length_slider = gr.Slider(
                minimum=100, maximum=700, value=200, step=20,
                label="Article Length", info="Number of tokens (words/phrases)."
            )
            temperature_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                label="Creativity (Temperature)",
                info="Lower = more factual, Higher = more creative."
            )
            style_selector = gr.Dropdown(
                label="Writing Style",
                choices=list(GEN_STYLES.keys()),
                value="Neutral",
                info="Select the tone/style of the generated news."
            )
            with gr.Row():
                generate_btn = gr.Button("🚀 Generate News", elem_id="gen-btn")
                sample1 = gr.Button("Try Sample", elem_classes="sample-btn")
            output_text = gr.Textbox(
                label="Generated News Article",
                interactive=False,
                lines=10,
                show_copy_button=True,
                placeholder="Your AI-generated news article will appear here!"
            )

            def use_sample():
                return "Scientists discover city-sized diamond beneath Africa"

            sample1.click(fn=use_sample, inputs=[], outputs=input_text, scroll_to_output=True)
            generate_btn.click(
                generate_fake_news,
                inputs=[input_text, max_length_slider, temperature_slider, style_selector, lang_selector],
                outputs=output_text,
                api_name="generate_news"
            )

        with gr.Column(scale=1, min_width=350):
            gr.Markdown(
                """
                <h3>🔍 Fake News Detector</h3>
                <span style="color:#363636">Paste news, and multilingual BERT evaluates authenticity.</span>
                """
            )
            detect_input = gr.Textbox(
                label="Input News Article",
                placeholder="Paste news content or statement here...",
                lines=8,
                info="Longer, complete paragraphs improve result accuracy."
            )
            with gr.Row():
                detect_btn = gr.Button("Analyze", elem_id="det-btn")
                sample2 = gr.Button("Try Example", elem_classes="sample-btn")
            detect_output = gr.Markdown(
                value="Detector results will be shown here.",
                label="Detection Result"
            )
            confidence_chart = gr.Plot()

            def detector_sample():
                return (
                    "The United Nations confirmed today that energy production on "
                    "the Moon could supply Earth within 15 years, making fossil fuels obsolete."
                )

            sample2.click(detector_sample, inputs=[], outputs=detect_input, scroll_to_output=True)

            def detect_and_chart(text, lang):
                res, conf_data = detect_news(text, lang)
                chart = make_confidence_bar_chart(conf_data)
                return res, chart

            detect_btn.click(detect_and_chart, inputs=[detect_input, lang_selector], outputs=[detect_output, confidence_chart])

    gr.Markdown(
        """
        <div style="text-align:center; color:#888; margin-top:12px;">
        ⚡ <b>Note:</b> This app is for demo & research.<br>
        All generated articles are synthetic.
        </div>
        """
    )

demo.launch()
