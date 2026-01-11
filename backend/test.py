# ==========================================
# TEST VNCORENLP (OFFICIAL – STABLE)
# ==========================================

from vncorenlp import VnCoreNLP

VNCORENLP_JAR = r"C:/Users/84913/OneDrive/Documents/Tai_lieu/NCKH/Demo/models/vncorenlp_model/VnCoreNLP-1.2.jar"

print("🔄 Loading VnCoreNLP (official)...")

vncorenlp = VnCoreNLP(
    VNCORENLP_JAR,
    annotators="wseg",
    max_heap_size='-Xmx2g'
)

print("✅ VnCoreNLP loaded successfully")


def tokenize(text: str):
    """
    Output: ['em_gái', 'đôi', 'mắt', 'đẹp']
    """
    sentences = vncorenlp.tokenize(text)

    tokens = []
    for sent in sentences:
        for tok in sent:
            tokens.append(tok.replace(" ", "_").lower())

    return tokens


if __name__ == "__main__":
    tests = [
        "em gái",
        "em gái đôi mắt đẹp",
        "tôi yêu ngôn ngữ ký hiệu",
        "cô ấy đang học đại học"
    ]

    for t in tests:
        print("\nInput :", t)
        print("Tokens:", tokenize(t))

    print("\n🎉 TEST OK – VNCORENLP OFFICIAL RUNNING")
