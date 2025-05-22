import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def predict_text_toxicity(text: str,
                          best_model_path: str = "../models/best_model/",
                          tokenizer_path: str = "../models/tokenizer/") -> int:
    """
    Predict whether the input text is toxic or not using a fine-tuned transformer model.

    Args:
        text (str): The input text to classify.
        best_model_path (str): Path to the model architecture directory (transformers-compatible).
        tokenizer_path (str): Path to the tokenizer directory.

    Returns:
        int: 1 if the text is predicted as toxic, 0 if non-toxic.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)
    model.eval()

    # Tokenize input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Run prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class
