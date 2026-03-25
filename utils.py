def skill_tokenizer(text):
    return [s.strip().lower() for s in text.split(",")]