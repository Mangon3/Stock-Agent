import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import time 
import random 

from src.utils.logger import setup_logger
logger = setup_logger(__name__)

from src.config.settings import settings
from src.tools.model.data import tv_data_fetcher
from src.tools.model.neural import GRU_StockNet
from src.tools.model.feature import FeatureCalculator
from src.utils.device import get_device

# --- Data Loading Class (Remains outside the Trainer) ---
class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for handling time-series sequence data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Convert NumPy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def create_sequences(df: pd.DataFrame, seq_len: int, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        
        if 'close' not in df.columns:
             raise ValueError("DataFrame must contain a 'close' column to calculate the target.")

        # Binary classification target: 1 if next day's close > current day's close
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop the last row where the target is NaN
        df = df.iloc[:-1]
        
        # Ensure all features and the target are present and not NaN
        df = df.dropna(subset=feature_cols + ['target'])
        
        feature_data = df[feature_cols].values
        target_data = df['target'].values
        
        for i in range(len(feature_data) - seq_len):
            seq = feature_data[i : (i + seq_len)]
            # Target is the label corresponding to the *last day* in the sequence
            label = target_data[i + seq_len - 1] 
            X.append(seq)
            y.append(label)

        if not X:
            logger.warning(f"Could not create any sequences. Data length: {len(df)}, SEQ_LEN: {seq_len}")
            return np.array([]), np.array([])
            
        return np.array(X), np.array(y)


# --- Model Trainer Class ---

class StockModelTrainer:

    
    # Default parameters, which can be overridden by the caller (micro.py tool)
    DEFAULT_SYMBOLS: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    DEFAULT_EPOCHS: int = 50
    DEFAULT_BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    TEST_SIZE_RATIO: float = 0.2
    DEVICE = get_device()

    def __init__(self):
        self.best_test_accuracy = 0.0
        logger.info(f"Trainer initialized. Target device: {self.DEVICE}.")

    def _train_test_split(self, X: np.ndarray, y: np.ndarray, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits the combined data into training and testing sets."""
        if len(X) != len(y):
            raise ValueError("Feature array X and target array y must have the same length.")

        test_size = int(len(X) * test_ratio)
        
        if test_size == 0 or test_size >= len(X):
            logger.warning("Test size is too small or too large. Using default minimum split.")
            test_size = max(1, min(len(X) // 5, len(X) - 1))

        split_index = len(X) - test_size
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        logger.info(f"Split data: Train samples={len(X_train)}, Test samples={len(X_test)}")
        
        return X_train, X_test, y_train, y_test


    def _load_and_prepare_data(self, symbols: List[str]) -> Tuple[StockDataset, StockDataset]:
        """Fetches data, calculates features, and creates the final sequence datasets."""
        
        all_X, all_y = [], []
        
        logger.info(f"Starting data fetch and feature engineering for {len(symbols)} symbols...")
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            
            df_raw = tv_data_fetcher.fetch_historical_data(
                symbol, 
                timeframe_days=settings.DATA_TIMEFRAME_DAYS, 
                exchange="NASDAQ"
            )

            if isinstance(df_raw, dict) and "error" in df_raw:
                logger.error(f"Skipping {symbol}: Data fetch failed: {df_raw['error']}")
                time.sleep(random.uniform(1, 3)) # Shorter sleep for tool execution
                continue
                
            try:
                # 1. Calculate features using the clean, external calculator
                df_features = FeatureCalculator.calculate_features(df_raw.copy())
                # 2. Add the original 'close' price back for target calculation
                df_features['close'] = df_raw['close']

                # 3. Create sequences
                X, y = StockDataset.create_sequences(
                    df_features,
                    settings.SEQ_LEN, 
                    settings.FEATURE_COLUMNS
                )
                
                if X.size > 0:
                    all_X.append(X)
                    all_y.append(y)
                    logger.info(f"Success for {symbol}: Created {X.shape[0]} sequences.")
                else:
                    logger.warning(f"Skipping {symbol}: Insufficient data after sequence creation.")

            except ValueError as e:
                logger.error(f"Skipping {symbol} due to feature calculation error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred for {symbol}: {str(e)}")


        if not all_X:
            raise RuntimeError("No training data could be generated for any symbol.")

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Split data and return PyTorch Datasets
        X_train, X_test, y_train, y_test = self._train_test_split(
            X_combined, 
            y_combined, 
            self.TEST_SIZE_RATIO
        )
            
        return StockDataset(X_train, y_train), StockDataset(X_test, y_test)


    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluates loss and accuracy on a given dataset (train or test)."""
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()
                
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = running_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy


    def train(self, symbols: List[str] = None, num_epochs: int = None, batch_size: int = None) -> Dict[str, Any]:
        logger.info(f"Trainer.train started. Device: {self.DEVICE}")
        """
        The main public method to start and run the model training process.
        
        Args:
            symbols (List[str], optional): List of stock symbols to train on. 
                Defaults to internal list if None.
            num_epochs (int, optional): Number of training epochs. Defaults to 50 if None.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32 if None.

        Returns:
            Dict[str, Any]: Final training metrics and status.
        """
        
        symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS
        num_epochs = num_epochs if num_epochs is not None else self.DEFAULT_EPOCHS
        batch_size = batch_size if batch_size is not None else self.DEFAULT_BATCH_SIZE
        
        try:
            # 1. Load and Prepare Data
            train_dataset, test_dataset = self._load_and_prepare_data(symbols)
            
            # Setup DataLoaders
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"\nTotal training samples available: {len(train_dataset)}")
            logger.info(f"Total testing samples available: {len(test_dataset)}")

            # Validation check
            if train_dataset.X.shape[2] != settings.INPUT_SIZE:
                 raise RuntimeError(f"Feature count mismatch! Expected {settings.INPUT_SIZE}, got {train_dataset.X.shape[2]}.")

            # 2. Initialize Model and Optimizers
            model = GRU_StockNet(
                input_size=settings.INPUT_SIZE,
                hidden_dim=settings.HIDDEN_DIM,
                num_layers=settings.NUM_LAYERS,
                dropout=settings.DROPOUT
            ).to(self.DEVICE)
            
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)

            self.best_test_accuracy = 0.0

            logger.info(f"Training started for {num_epochs} epochs on symbols: {', '.join(symbols)}")
            
            # 3. Training Loop
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                
                # Training step
                for i, (inputs, labels) in enumerate(train_dataloader):
                    inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                
                train_loss = running_loss / len(train_dataloader)
                
                # Evaluation step
                test_loss, test_accuracy = self._evaluate_model(model, test_dataloader)

                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
                
                # Checkpointing
                if test_accuracy > self.best_test_accuracy:
                    self.best_test_accuracy = test_accuracy
                    
                    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), settings.MODEL_PATH)
                    logger.info(f"--> Model saved! New BEST Test Accuracy: {self.best_test_accuracy:.4f}")
                
            logger.info("Training complete.")
            
            return {
                "status": "success",
                "final_accuracy": self.best_test_accuracy,
                "epochs_run": num_epochs,
                "model_path": str(settings.MODEL_PATH)
            }

        except RuntimeError as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": f"Training failed: {e}"}
        except Exception as e:
            logger.error(f"An unexpected error occurred during training: {e}")
            return {"status": "error", "message": f"Unexpected error: {e}"}

trainer = StockModelTrainer()