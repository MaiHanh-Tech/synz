from translate_book import translate_file, create_interactive_html_block
from html import escape

def translate_document(text, progress_callback=None, include_english=True, source_lang="Chinese", target_lang="Vietnamese", mode="Standard Translation", processed_words=None):
    """
    Thin wrapper around translate_book.translate_file to keep orchestrator layer.
    """
    return translate_file(text, status_placeholder=None, progress_bar=progress_callback, include_english=include_english, source_lang=source_lang, target_lang=target_lang, translation_mode=mode, processed_words=processed_words)
