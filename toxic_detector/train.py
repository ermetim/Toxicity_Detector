import torch
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra
from datasets import Dataset
from transformers import DistilBertTokenizer

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer import LitDistilBERT


# Функция для преобразования батча
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': labels}


# Внешняя функция для создания DataLoader
def create_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


# Функция для токенизации
def tokenize_function(tokenizer, examples, max_length):
    return tokenizer(examples['comment_text'], padding="max_length", truncation=True, max_length=max_length)


# Основная функция Hydra
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    tokenizer = DistilBertTokenizer.from_pretrained(config.model.tokenizer)

    # Загрузка данных из CSV
    df = pd.read_csv(config.data_loading.data_path)

    # Разбиваем данные на тренировочную и валидационную выборки
    train_df = df.sample(frac=config.data_loading.train_data_fraction, random_state=config.data_loading.random_state)
    val_df = df.drop(train_df.index)

    # Преобразуем pandas DataFrame в формат, который принимает HuggingFace Datasets
    train_data = Dataset.from_pandas(train_df)
    val_data = Dataset.from_pandas(val_df)

    # Токенизация
    train_data = train_data.map(
        lambda examples: tokenize_function(tokenizer, examples, config.model.max_length),
        batched=True
    )
    val_data = val_data.map(
        lambda examples: tokenize_function(tokenizer, examples, config.model.max_length),
        batched=True
    )

    # Переименование меток
    train_data = train_data.rename_column(config.data_loading.label_column, "label")
    val_data = val_data.rename_column(config.data_loading.label_column, "label")

    # Создание DataLoader
    train_loader = create_dataloader(
        train_data,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers
    )

    # Инициализация модели
    model = LitDistilBERT(model_name=config.model.pretrained_model, learning_rate=config.training.learning_rate)

    # Инициализация ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.model.model_local_path,
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # Настройка логирования (например, TensorBoardLogger)
    logger = TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name=config.logging.name
    )

    # Инициализация тренера
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=config.training.log_every_n_steps,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Запуск обучения
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
