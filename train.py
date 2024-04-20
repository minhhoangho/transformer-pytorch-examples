
from models.model.bert_torch import BertCRF
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torch.optim import Adam

def train(dataset, num_epochs=10):

    n_tag = 3
    max_len = 384
    output_dim = 512

    # Initialize model, tokenizer, and optimizer
    model = BertCRF(output_dim=output_dim, max_len=max_len, n_tags=n_tag)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    optimizer = Adam(model.parameters())

    # Assume you have a Dataset object `dataset`
    dataloader = DataLoader(dataset, batch_size=32)

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Unpack batch
            input_ids, token_type_ids, attention_mask, labels = batch

            # Tokenize inputs
            input_ids = tokenizer(input_ids, padding=True, truncation=True, max_length=model.max_len,
                                  return_tensors='pt')
            token_type_ids = tokenizer(token_type_ids, padding=True, truncation=True, max_length=model.max_len,
                                       return_tensors='pt')
            attention_mask = tokenizer(attention_mask, padding=True, truncation=True, max_length=model.max_len,
                                       return_tensors='pt')

            # Forward pass
            outputs = model(input_ids, token_type_ids, attention_mask)
            predicted_labels = outputs[0]
            loss = model.get_loss(input_ids, token_type_ids, attention_mask, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Print loss and accuracy for this epoch
        print(
            f'Epoch {epoch}, Loss: {loss.item()}, Accuracy: {model.get_accuracy(input_ids, token_type_ids, attention_mask, labels)}')

if __name__ == "__main__":
    dataset = Dataset()
    train(dataset, num_epochs=10)

