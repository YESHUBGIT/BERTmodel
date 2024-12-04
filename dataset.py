import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        # Ensure truncation and padding
        inputs = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',  # Return tensors directly
        )

        # Extract tokenized tensors and flatten to remove extra dimensions
        ids = inputs["input_ids"].squeeze(0)  # Remove the extra batch dimension (1)
        mask = inputs["attention_mask"].squeeze(0)
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(ids)).squeeze(0)

        return {
            "ids": ids,
            "mask": mask,
            "token_type_ids": token_type_ids,
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }