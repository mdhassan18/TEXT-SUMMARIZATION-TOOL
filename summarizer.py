import os
import textwrap
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def display_banner():
    print("=" * 80)
    print(" " * 25 + "🧠 ARTICLE SUMMARIZER TOOL 📝")
    print("=" * 80)

def wrap_text(text, width=80):
    return "\n".join(textwrap.wrap(text, width=width))

def load_text_from_file(file_path):
    """Read and return the contents of a .txt file"""
    if not os.path.exists(file_path):
        print("❌ Error: File not found.")
        return None
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def choose_model():
    """Allow user to choose summarization model"""
    print("\nSelect summarization model:")
    print("1. BART (facebook/bart-large-cnn) [Recommended]")
    print("2. T5 (t5-small)")
    choice = input("Enter choice [1/2]: ").strip()
    if choice == "2":
        print("🔁 Loading T5-small...")
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    else:
        print("🔁 Loading BART-large-cnn...")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("✅ Model loaded successfully!")
    return summarizer

def summarize_text(summarizer, text, min_len=30, max_len=130):
    """Perform summarization using the chosen model"""
    try:
        summary_output = summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summary_output[0]['summary_text']
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return ""

def main():
    display_banner()

    print("\nHow would you like to input the article?")
    print("1. Paste article manually")
    print("2. Load article from .txt file")
    input_choice = input("Enter choice [1/2]: ").strip()

    if input_choice == "1":
        print("\n📥 Paste or type your article below (End input with Ctrl+D or Ctrl+Z):\n")
        print(">>>")
        article_text = ""
        try:
            while True:
                line = input()
                article_text += line + "\n"
        except EOFError:
            pass
    elif input_choice == "2":
        file_path = input("\nEnter full path to the .txt file: ").strip()
        article_text = load_text_from_file(file_path)
        if article_text is None:
            return
    else:
        print("❌ Invalid input choice.")
        return

    if not article_text.strip():
        print("❌ No content found to summarize.")
        return

    summarizer = choose_model()

    print("\n⏳ Summarizing the article... Please wait.\n")
    summary = summarize_text(summarizer, article_text)

    print("\n" + "=" * 80)
    print("📜 ORIGINAL ARTICLE (truncated):")
    print("=" * 80)
    print(wrap_text(article_text[:1000]))  # Displaying only first 1000 characters

    print("\n" + "=" * 80)
    print("🧾 GENERATED SUMMARY:")
    print("=" * 80)
    print(wrap_text(summary))

    print("\n✅ Done. Thank you for using the summarizer tool!")

if __name__ == "__main__":
    main()
input("\nPress Enter to exit...")
