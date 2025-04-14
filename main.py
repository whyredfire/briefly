import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

text = """
The global climate crisis is one of the most significant challenges humanity faces today. Rising temperatures, extreme weather events, and disruptions to ecosystems have far-reaching consequences for life on Earth. Governments, scientists, and organizations are working together to find solutions to mitigate climate change. Renewable energy sources like solar and wind power are becoming more common, and there is a global push to reduce carbon emissions. Despite these efforts, many regions still face challenges in adapting to the changing climate. Immediate action is needed to prevent further damage to the planet's environment.
"""

# tokenize sentences
sentences = sent_tokenize(text)

# tokenize words and convert to lower case
words = word_tokenize(text.lower())

# count word frequencies (ignore stopwords and non-alphanumeric words)
freq = {}
for word in words:
    if word.isalnum() and word not in stopwords.words("english"):
        freq[word] = freq.get(word, 0) + 1

# normalize frequencies (so the frequencies are between 0 and 1)
max_freq = max(freq.values(), default=1)
freq = {w: f / max_freq for w, f in freq.items()}

# score each sentence
scores = {}
for sent in sentences:
    for word in word_tokenize(sent.lower()):
        if word in freq:
            scores[sent] = scores.get(sent, 0) + freq[word]

# pick top sentences based on score (limit to 0.3 times of input text)
k = max(1, int(len(sentences) * 0.3))
top = sorted(scores, key=scores.get, reverse=True)[:k]

print("Summary:\n", " ".join(top))

