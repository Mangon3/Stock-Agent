import torch
torch.set_num_threads(1)
import pandas as pd
from typing import Dict, Any, Tuple

from src.config.settings import settings
from src.tools.model.data import tv_data_fetcher
from src.tools.model.neural import GRU_StockNet
from src.tools.model.feature import features
from src.utils.device import get_device


def inference_results(prob: float) -> Tuple[str, str]:
    """Interprets the model's output probability into a human-readable signal and outlook."""
    if prob > 0.60:
        signal = "STRONG_BULLISH"
        outlook = "High confidence for a strong upward move (Strong Buy Signal)"
    elif prob > 0.52:
        signal = "BULLISH"
        outlook = "Positive Momentum (Buy Signal)"
    elif prob < 0.40:
        signal = "STRONG_BEARISH"
        outlook = "High confidence for a strong downward move (Strong Sell Signal)"
    elif prob < 0.48:
        signal = "BEARISH"
        outlook = "Negative Momentum (Sell Signal)"
    else:
        signal = "NEUTRAL"
        outlook = "Sideways / Low Confidence"
    return signal, outlook

class MicroModelPredictor:

    def __init__(self):
        self.data_fetcher = tv_data_fetcher 
        pass

    def _load_model(self) -> GRU_StockNet:
        if not settings.MODEL_PATH.exists():
            # If the model doesn't exist, we must raise an error to indicate training is required
            raise FileNotFoundError(f"Trained model file not found at: {settings.MODEL_PATH}. Please run training first.")

        model = GRU_StockNet(
            input_size=settings.INPUT_SIZE,
            hidden_dim=settings.HIDDEN_DIM,
            num_layers=settings.NUM_LAYERS,
            dropout=settings.DROPOUT
        )
        
        # Load the saved state dict
        device = get_device()
        model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        
        return model

    def _prepare_data_for_inference(self, df: pd.DataFrame) -> torch.Tensor:
        # Assuming features.calculate_features is accessible and functional
        feature_df = features.calculate_features(df)
        
        if len(feature_df) < settings.SEQ_LEN:
             raise ValueError(f"Insufficient data points ({len(feature_df)}) for sequence length ({settings.SEQ_LEN}). Need at least {settings.SEQ_LEN} sequential feature rows.")

        feature_data = feature_df[settings.FEATURE_COLUMNS]
        sequence_data = feature_data.tail(settings.SEQ_LEN).values
        input_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
        
        return input_tensor

    def predict_price_outlook(self, symbol: str, exchange: str = "NASDAQ") -> Dict[str, Any]:
        try:
            model = self._load_model()
        except FileNotFoundError as e:
            return {"error": str(e)}

        df = self.data_fetcher.fetch_historical_data(
            symbol, 
            timeframe_days=settings.DATA_TIMEFRAME_DAYS, 
            exchange=exchange
        )

        if isinstance(df, dict) and "error" in df:
            return {"error": f"Data fetch failed for {symbol}: {df['error']}"}

        try:
            input_tensor = self._prepare_data_for_inference(df)
            
            with torch.no_grad():
                logits = model(input_tensor) # Use the locally loaded model
                probability = torch.sigmoid(logits).item()

            signal, outlook_text = inference_results(probability)
            
            latest_price = df['close'].iloc[-1]

            return {
                "symbol": symbol.upper(),
                "latest_close_price": float(f"{latest_price:.2f}"),
                "model_output_probability": float(f"{probability:.4f}"),
                "signal": signal,
                "outlook": outlook_text,
                "confidence_level": f"{abs(probability - 0.5) * 2 * 100:.1f}%"
            }

        except Exception as e:
            return {"error": f"Model inference failed for {symbol}. Reason: {e}"}

micro_model_predictor = MicroModelPredictor()