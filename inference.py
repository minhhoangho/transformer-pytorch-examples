from models.model.bert_torch import BertCRF
import torch


def inference(sentences: list[str]):
    from transformers import BertTokenizer

    # Initialize model and tokenizer
    model = BertCRF(num_tags=10, max_len=512)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load trained model weights
    model.load_state_dict(torch.load('model_weights.pth'))

    # Assume you have a list of sentences `sentences`
    for sentence in sentences:
        # Tokenize input
        inputs = tokenizer(sentence, padding=True, truncation=True, max_length=model.max_len, return_tensors='pt')

        # Forward pass
        outputs = model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        predicted_labels = outputs[0]

        # Print predicted labels
        print(predicted_labels)
