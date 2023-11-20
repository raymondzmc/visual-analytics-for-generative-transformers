from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pdb


def bert_classifier():
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    setattr(model, "num_hidden_layers", model.config.num_hidden_layers)
    setattr(model, "num_attention_heads", model.config.num_attention_heads)
    setattr(model, "hidden_size", model.config.hidden_size)
    return model
