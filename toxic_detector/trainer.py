import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification, AdamW


class LitDistilBERT(pl.LightningModule):
    def __init__(self, model_name, learning_rate=5e-5):
        super(LitDistilBERT, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)
