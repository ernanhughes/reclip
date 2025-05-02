
class SimilarityJudge:
    def __init__(self, strong_threshold=0.6, weak_threshold=0.3):
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

    def evaluate(self, score: float) -> str:
        if score >= self.strong_threshold:
            return "✅ Strong Match"
        elif score >= self.weak_threshold:
            return "🟡 Moderate Match"
        elif score >= 0.0:
            return "⚠️ Weak Match"
        else:
            return "❌ Mismatch"
