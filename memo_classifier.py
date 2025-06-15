class MemoClassifier:
    """Classify text sentiment using NLTK's VADER."""

    def __init__(self):
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        # Attempt to load VADER lexicon; download if missing
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            import nltk
            nltk.download('vader_lexicon')
            self.analyzer = SentimentIntensityAnalyzer()

    def classify(self, text: str) -> str:
        """Return 'positive' if compound score >= 0, else 'negative'."""
        scores = self.analyzer.polarity_scores(text)
        return 'positive' if scores['compound'] >= 0 else 'negative'


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python memo_classifier.py \"your text\"")
        sys.exit(1)

    text = sys.argv[1]
    classifier = MemoClassifier()
    result = classifier.classify(text)
    print(result)
