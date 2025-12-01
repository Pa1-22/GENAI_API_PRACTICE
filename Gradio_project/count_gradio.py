import gradio as gr

def analyze_text(text):
    text = text.strip()
    num_chars = len(text)
    num_words = len(text.split()) if text else 0
    
    # Make output more readable
    summary = f"✅ Characters: {num_chars}\n✅ Words: {num_words}"
    return summary

demo = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type your text here...",
        label="Enter some text"
    ),
    outputs=gr.Textbox(
        lines=3,             # Make output box bigger
        label="Analysis",
        interactive=False     # User cannot edit output
    ),
    title="Simple Text Analyzer",
    description="Counts characters and words. You can replace the function with any ML model later."
)

if __name__ == "__main__":
    demo.launch()
