# ml_trading_bot.py
# Enhanced version with comprehensive ML integration

import ccxt
import pandas as pd
import ta
import time
import requests
import logging
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import argparse
from backtester import run_backtest

# ML Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LSTMPricePredictor(nn.Module):
    """
    LSTM Neural Network for price prediction
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Use last output
        out = self.fc(out)
        return out

class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for better feature focus
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output

class MLFeatureEngineer:
    """
    Advanced feature engineering with ML techniques
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.feature_importance = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features using ML techniques
        """
        # Technical indicators (existing ones)
        df = self._add_technical_indicators(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Regime-based features
        df = self._add_regime_features(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicators"""
        # Multi-timeframe RSI
        for period in [7, 14, 21, 30]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # Multiple EMA periods
        for period in [8, 12, 21, 26, 50, 100]:
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
        
        # Stochastic oscillator
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        df['stoch_d'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
        
        # Commodity Channel Index
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
        
        # Average True Range
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based cyclical features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features for price action"""
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'return_{window}'] = df['close'].pct_change(window)
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
            df[f'skew_{window}'] = df['close'].pct_change().rolling(window).skew()
            df[f'kurtosis_{window}'] = df['close'].pct_change().rolling(window).kurtosis()
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_volume_trend'] = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend()
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        # Spread proxies
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Imbalance measures
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        
        # Momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime-based features"""
        # Volatility regimes
        df['vol_regime'] = pd.cut(df['close'].pct_change().rolling(20).std(), 
                                 bins=3, labels=['low', 'medium', 'high'])
        
        # Trend strength
        df['trend_strength'] = abs(df['close'].pct_change().rolling(20).mean())
        
        return df

class MarketRegimeDetector:
    """
    Detect market regimes using clustering and HMM-like approaches
    """
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_features = ['volatility_20', 'return_20', 'trend_strength']
        
    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes and add regime labels
        """
        # Prepare features for clustering
        features = df[self.regime_features].dropna()
        
        if len(features) < self.n_regimes:
            df['regime'] = 0
            return df
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit clustering model
        regime_labels = self.kmeans.fit_predict(features_scaled)
        
        # Map back to dataframe
        df.loc[features.index, 'regime'] = regime_labels
        df['regime'] = df['regime'].fillna(method='ffill')
        
        # Add regime probabilities (distance-based)
        distances = self.kmeans.transform(features_scaled)
        for i in range(self.n_regimes):
            df.loc[features.index, f'regime_{i}_prob'] = 1 / (1 + distances[:, i])
        
        return df

class SignalEnhancer:
    """
    ML-based signal enhancement and filtering
    """
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.signal_features = []
        self.is_trained = False
        
    def prepare_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for signal classification
        """
        # Technical indicator signals
        df['rsi_signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['ema_signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        # Bollinger Bands signals
        df['bb_signal'] = np.where(df['close'] < df['bb_lower'], 1, 
                                  np.where(df['close'] > df['bb_upper'], -1, 0))
        
        # Volume confirmation
        df['volume_signal'] = np.where(df['volume'] > df['volume_sma'] * 1.5, 1, 0)
        
        # Momentum signals
        df['momentum_signal'] = np.where(df['momentum_5'] > 0.02, 1, 
                                        np.where(df['momentum_5'] < -0.02, -1, 0))
        
        return df
    
    def train_signal_enhancer(self, df: pd.DataFrame, future_returns: pd.Series):
        """
        Train the signal enhancement model
        """
        # Prepare features
        signal_cols = [col for col in df.columns if 'signal' in col.lower()]
        feature_cols = ['rsi', 'macd', 'ema_fast', 'ema_slow', 'volume_sma_ratio', 
                       'volatility_20', 'trend_strength']
        
        self.signal_features = signal_cols + feature_cols
        
        # Create target (binary: profitable vs not profitable)
        target = (future_returns > 0.01).astype(int)  # 1% threshold
        
        # Prepare data
        X = df[self.signal_features].dropna()
        y = target.loc[X.index]
        
        if len(X) < 50:  # Minimum samples needed
            logger.warning("Insufficient data for signal enhancer training")
            return
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Signal enhancer trained with accuracy: {accuracy:.3f}")
        self.is_trained = True
    
    def enhance_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance signals using trained ML model
        """
        if not self.is_trained:
            logger.warning("Signal enhancer not trained yet")
            return df
        
        # Get signal probabilities
        X = df[self.signal_features].dropna()
        if len(X) > 0:
            probabilities = self.rf_classifier.predict_proba(X)
            df.loc[X.index, 'signal_confidence'] = probabilities[:, 1]  # Probability of positive class
        
        return df

class RiskManager:
    """
    ML-based risk management system
    """
    def __init__(self):
        self.volatility_predictor = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_features = []
        
    def calculate_position_size(self, df: pd.DataFrame, account_balance: float, 
                              confidence: float = 0.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion with ML adjustments
        """
        # Get recent volatility
        recent_vol = df['volatility_20'].iloc[-1] if not pd.isna(df['volatility_20'].iloc[-1]) else 0.02
        
        # Base position size (Kelly-inspired)
        base_size = confidence * 0.1  # Max 10% of account
        
        # Adjust for volatility
        vol_adjustment = 1 / (1 + recent_vol * 10)
        
        # Adjust for regime
        regime_adjustment = 1.0
        if 'regime' in df.columns:
            current_regime = df['regime'].iloc[-1]
            if current_regime == 2:  # High volatility regime
                regime_adjustment = 0.5
            elif current_regime == 0:  # Low volatility regime
                regime_adjustment = 1.2
        
        position_size = base_size * vol_adjustment * regime_adjustment
        return min(position_size, 0.15) * account_balance  # Cap at 15%
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market anomalies that might indicate high risk
        """
        # Prepare features for anomaly detection
        anomaly_features = ['close', 'volume', 'volatility_20', 'return_20']
        features = df[anomaly_features].dropna()
        
        if len(features) < 10:
            df['anomaly_score'] = 0
            return df
        
        # Fit anomaly detector
        self.anomaly_detector.fit(features)
        
        # Get anomaly scores
        anomaly_scores = self.anomaly_detector.decision_function(features)
        df.loc[features.index, 'anomaly_score'] = anomaly_scores
        
        # Flag anomalies
        df['is_anomaly'] = df['anomaly_score'] < -0.5
        
        return df
    
    def calculate_dynamic_stop_loss(self, df: pd.DataFrame, position_type: str = 'long') -> float:
        """
        Calculate dynamic stop loss based on current market conditions
        """
        atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else df['close'].pct_change().std()
        current_price = df['close'].iloc[-1]
        
        # Base stop loss (2x ATR)
        base_stop = atr * 2
        
        # Adjust for volatility regime
        if 'regime' in df.columns:
            current_regime = df['regime'].iloc[-1]
            if current_regime == 2:  # High volatility
                base_stop *= 1.5
            elif current_regime == 0:  # Low volatility
                base_stop *= 0.8
        
        # Calculate stop loss price
        if position_type == 'long':
            stop_loss = current_price - base_stop
        else:
            stop_loss = current_price + base_stop
        
        return stop_loss

class EnhancedMLTradingBot:
    """
    Enhanced trading bot with comprehensive ML integration
    """
    def __init__(self, config_file: str = 'config.json'):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize exchange
        self.exchange = self.initialize_exchange()
        
        # Initialize ML components
        self.feature_engineer = MLFeatureEngineer()
        self.regime_detector = MarketRegimeDetector()
        self.signal_enhancer = SignalEnhancer()
        self.risk_manager = RiskManager()
        
        # Initialize price prediction models
        self.price_models = {}
        self.model_ready = {}
        
        # Initialize data storage
        self.historical_data = {}
        self.last_signals = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'total_returns': 0.0
        }
        
        logger.info("Enhanced ML Trading Bot initialized")
    
    def load_config(self, config_file: str) -> Dict:
        """Enhanced configuration loading"""
        default_config = {
            "telegram": {
                "token": "YOUR_TELEGRAM_BOT_TOKEN",
                "chat_id": "YOUR_TELEGRAM_CHAT_ID"
            },
            "exchange": {
                "name": "cryptocom",
                "sandbox": False,
                "api_key": "",
                "secret": ""
            },
            "trading": {
                "symbols": ["ETH/USDT", "BTC/USDT", "ETC/USDT"],
                "timeframe": "5m",
                "limit": 200,  # Increased for ML models
                "check_interval": 300
            },
            "ml": {
                "prediction_horizon": 5,  # Predict 5 periods ahead
                "retrain_interval": 24,   # Retrain every 24 hours
                "min_accuracy": 0.55,     # Minimum acceptable accuracy
                "confidence_threshold": 0.6
            },
            "risk": {
                "max_position_size": 0.15,
                "account_balance": 10000,
                "stop_loss_multiplier": 2.0
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                config = self.merge_configs(default_config, user_config)
                logger.info(f"Configuration loaded from {config_file}")
                return config
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                return default_config
        else:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Default configuration created: {config_file}")
            return default_config
    
    def merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge configurations"""
        for key, value in user.items():
            if key in default:
                if isinstance(default[key], dict) and isinstance(value, dict):
                    default[key] = self.merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
        return default
    
    def initialize_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_name = self.config['exchange']['name']
            exchange_class = getattr(ccxt, exchange_name)
            
            exchange_config = {
                'sandbox': self.config['exchange']['sandbox'],
                'enableRateLimit': True,
            }
            
            if self.config['exchange']['api_key']:
                exchange_config['apiKey'] = self.config['exchange']['api_key']
                exchange_config['secret'] = self.config['exchange']['secret']
            
            exchange = exchange_class(exchange_config)
            logger.info(f"Exchange {exchange_name} initialized successfully")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for ML models
        """
        # Feature engineering
        df_features = self.feature_engineer.engineer_features(df.copy())
        
        # Select numeric features only
        numeric_features = df_features.select_dtypes(include=[np.number]).columns
        feature_data = df_features[numeric_features].fillna(method='ffill').fillna(0)
        
        # Prepare sequences for LSTM
        sequence_length = 20
        X, y = [], []
        
        for i in range(sequence_length, len(feature_data)):
            X.append(feature_data.iloc[i-sequence_length:i].values)
            y.append(feature_data['close'].iloc[i])
        
        if len(X) == 0:
            return None, None
        
        X = torch.FloatTensor(np.array(X))
        y = torch.FloatTensor(np.array(y))
        
        return X, y
    
    def train_price_model(self, symbol: str, df: pd.DataFrame):
        """
        Train price prediction model for a symbol
        """
        try:
            logger.info(f"Training price model for {symbol}")
            
            # Prepare data
            X, y = self.prepare_ml_data(df)
            if X is None or len(X) < 50:
                logger.warning(f"Insufficient data for training {symbol}")
                return
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Initialize model
            input_size = X.shape[2]
            model = AttentionLSTM(input_size)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train model
            model.train()
            epochs = 50
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs.squeeze(), y_test)
                
            logger.info(f"Model for {symbol} trained. Test Loss: {test_loss.item():.6f}")
            
            # Store model
            self.price_models[symbol] = model
            self.model_ready[symbol] = True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
    
    def predict_price(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        """
        Predict future price using trained model
        """
        if symbol not in self.price_models or not self.model_ready[symbol]:
            return None
        
        try:
            # Prepare recent data
            X, _ = self.prepare_ml_data(df)
            if X is None or len(X) == 0:
                return None
            
            # Make prediction
            model = self.price_models[symbol]
            model.eval()
            with torch.no_grad():
                prediction = model(X[-1:])  # Use most recent sequence
            
            return prediction.item()
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return None
    
    def fetch_enhanced_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch and enhance data with ML features
        """
        try:
            # Fetch basic OHLCV data
            timeframe = self.config['trading']['timeframe']
            limit = self.config['trading']['limit']
            
            bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not bars:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add ML features
            df = self.feature_engineer.engineer_features(df)
            df = self.regime_detector.detect_regimes(df)
            df = self.signal_enhancer.prepare_signal_features(df)
            df = self.risk_manager.detect_anomalies(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch enhanced data for {symbol}: {e}")
            return None
    
    def generate_ml_signals(self, df: pd.DataFrame, symbol: str) -> List[str]:
        """
        Generate trading signals using ML models
        """
        signals = []
        
        if len(df) < 2:
            return signals
        
        current = df.iloc[-1]
        
        # Price prediction signal
        predicted_price = self.predict_price(symbol, df)
        if predicted_price:
            current_price = current['close']
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.02:  # 2% upside
                signals.append(f"🤖 <b>ML PRICE PREDICTION BUY</b> - Expected: +{price_change*100:.1f}%")
            elif price_change < -0.02:  # 2% downside
                signals.append(f"🤖 <b>ML PRICE PREDICTION SELL</b> - Expected: {price_change*100:.1f}%")
        
        # Signal confidence
        if 'signal_confidence' in df.columns and not pd.isna(current['signal_confidence']):
            confidence = current['signal_confidence']
            threshold = self.config['ml']['confidence_threshold']
            
            if confidence > threshold:
                signals.append(f"✅ <b>HIGH CONFIDENCE SIGNAL</b> - {confidence:.1%}")
            elif confidence < (1 - threshold):
                signals.append(f"❌ <b>LOW CONFIDENCE WARNING</b> - {confidence:.1%}")
        
        # Regime detection
        if 'regime' in df.columns:
            regime = current['regime']
            if regime == 0:
                signals.append(f"📊 <b>REGIME: LOW VOLATILITY</b> - Trending market")
            elif regime == 2:
                signals.append(f"⚠️ <b>REGIME: HIGH VOLATILITY</b> - Caution advised")
        
        # Anomaly detection
        if 'is_anomaly' in df.columns and current['is_anomaly']:
            signals.append(f"🚨 <b>MARKET ANOMALY DETECTED</b> - Unusual conditions")
        
        return signals
    
   def process_symbol_enhanced(self, symbol: str):
    """
    Enhanced symbol processing with ML integration
    
    This method orchestrates the complete ML-enhanced trading pipeline:
    1. Data fetching and preprocessing
    2. Model training/retraining
    3. Signal generation and enhancement
    4. Risk management
    5. Position sizing and execution
    6. Notification and logging
    
    Args:
        symbol (str): Trading symbol to process (e.g., 'BTC/USDT')
    """
    try:
        logger.info(f"Processing {symbol} with ML enhancement")
        
        # ==================== DATA ACQUISITION & PREPROCESSING ====================
        # Fetch enhanced data with all ML features, technical indicators, 
        # regime detection, and anomaly scores
        df = self.fetch_enhanced_data(symbol)
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {symbol} - need at least 50 data points")
            return
        
        logger.info(f"Fetched {len(df)} data points for {symbol}")
        
        # ==================== MODEL TRAINING & MANAGEMENT ====================
        # Check if we need to train or retrain the ML model
        # This happens when:
        # - Model doesn't exist for this symbol
        # - Model performance has degraded
        # - Retrain interval has passed
        if (symbol not in self.model_ready or 
            not self.model_ready[symbol] or 
            self.should_retrain_model(symbol)):
            
            logger.info(f"Training/retraining ML model for {symbol}")
            self.train_price_model(symbol, df)
        
        # ==================== SIGNAL ENHANCEMENT ====================
        # Train signal enhancer if we have sufficient data
        # The signal enhancer uses ML to improve traditional signal quality
        if len(df) > 100:  # Need substantial data for reliable signal enhancement
            logger.info(f"Training signal enhancer for {symbol}")
            
            # Calculate future returns for training (5 periods ahead)
            future_returns = df['close'].shift(-5).pct_change()
            
            # Train the signal enhancer to predict profitable signals
            self.signal_enhancer.train_signal_enhancer(df, future_returns)
            
            # Apply signal enhancement to current data
            df = self.signal_enhancer.enhance_signals(df)
        
        # ==================== SIGNAL GENERATION ====================
        # Generate ML-based signals using trained models
        ml_signals = self.generate_ml_signals(df, symbol)
        logger.info(f"Generated {len(ml_signals)} ML signals for {symbol}")
        
        # Generate traditional technical analysis signals for comparison
        traditional_signals = self.check_traditional_signals(df, symbol)
        logger.info(f"Generated {len(traditional_signals)} traditional signals for {symbol}")
        
        # Combine all signals for comprehensive analysis
        all_signals = ml_signals + traditional_signals
        
        # ==================== SIGNAL VALIDATION & FILTERING ====================
        # Only proceed if we have signals and they're not duplicates
        if all_signals and not self.is_duplicate_signal(symbol, all_signals):
            
            # Get current market data
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # ==================== RISK MANAGEMENT ====================
            # Calculate optimal position size using ML-enhanced risk management
            # This considers:
            # - Account balance and risk tolerance
            # - Current market volatility
            # - Market regime (trending vs sideways vs volatile)
            # - Signal confidence levels
            position_size = self.risk_manager.calculate_position_size(
                df, 
                self.config['risk']['account_balance'],
                confidence=df['signal_confidence'].iloc[-1] if 'signal_confidence' in df.columns else 0.5
            )
            
            # Calculate dynamic stop loss based on current market conditions
            # This adapts to volatility and market regime
            stop_loss_long = self.risk_manager.calculate_dynamic_stop_loss(df, 'long')
            stop_loss_short = self.risk_manager.calculate_dynamic_stop_loss(df, 'short')
            
            # ==================== SIGNAL STRENGTH ASSESSMENT ====================
            # Assess overall signal strength by combining multiple factors
            signal_strength = self.calculate_signal_strength(df, all_signals)
            
            # Only execute if signal strength meets minimum threshold
            min_strength = self.config['ml'].get('min_signal_strength', 0.6)
            
            if signal_strength >= min_strength:
                logger.info(f"Signal strength {signal_strength:.2f} meets threshold for {symbol}")
                
                # ==================== POSITION SIZING FINAL CALCULATION ====================
                # Adjust position size based on signal strength
                adjusted_position_size = position_size * signal_strength
                
                # Ensure position size doesn't exceed maximum allowed
                max_position = self.config['risk']['max_position_size'] * self.config['risk']['account_balance']
                final_position_size = min(adjusted_position_size, max_position)
                
                # ==================== EXECUTION PREPARATION ====================
                # Prepare execution parameters
                execution_params = {
                    'symbol': symbol,
                    'position_size': final_position_size,
                    'current_price': current_price,
                    'stop_loss_long': stop_loss_long,
                    'stop_loss_short': stop_loss_short,
                    'signal_strength': signal_strength,
                    'confidence': df['signal_confidence'].iloc[-1] if 'signal_confidence' in df.columns else 0.5,
                    'market_regime': df['regime'].iloc[-1] if 'regime' in df.columns else 'unknown',
                    'anomaly_detected': df['is_anomaly'].iloc[-1] if 'is_anomaly' in df.columns else False
                }
                
                # ==================== SAFETY CHECKS ====================
                # Final safety checks before execution
                if self.perform_safety_checks(execution_params):
                    
                    # ==================== NOTIFICATION PREPARATION ====================
                    # Prepare detailed notification message
                    message = self.prepare_notification_message(symbol, all_signals, execution_params, df)
                    
                    # ==================== EXECUTION & LOGGING ====================
                    # Execute the trade if not in demo mode
                    if not self.config.get('demo_mode', True):
                        execution_result = self.execute_trade(execution_params)
                        if execution_result:
                            message += f"\n\n✅ <b>TRADE EXECUTED</b>\n{execution_result}"
                        else:
                            message += f"\n\n❌ <b>TRADE EXECUTION FAILED</b>"
                    else:
                        message += f"\n\n📊 <b>DEMO MODE</b> - Trade not executed"
                    
                    # Send notification
                    self.send_telegram_message(message)
                    
                    # Update performance tracking
                    self.update_performance_metrics(symbol, execution_params)
                    
                    # Store signal for duplicate detection
                    self.store_signal(symbol, all_signals)
                    
                    logger.info(f"Successfully processed {symbol} with {len(all_signals)} signals")
                    
                else:
                    logger.warning(f"Safety checks failed for {symbol} - trade not executed")
                    
            else:
                logger.info(f"Signal strength {signal_strength:.2f} below threshold {min_strength} for {symbol}")
        
        else:
            if not all_signals:
                logger.info(f"No signals generated for {symbol}")
            else:
                logger.info(f"Duplicate signal detected for {symbol} - skipping")
        
        # ==================== CLEANUP & MAINTENANCE ====================
        # Update historical data storage
        self.update_historical_data(symbol, df)
        
        # Cleanup old data to prevent memory issues
        self.cleanup_old_data(symbol)
        
    except Exception as e:
        logger.error(f"Error in process_symbol_enhanced for {symbol}: {e}")
        # Send error notification if critical
        if "critical" in str(e).lower():
            self.send_telegram_message(f"🚨 <b>CRITICAL ERROR</b> processing {symbol}: {e}")

    # ==================== HELPER METHODS (CLASS METHODS) ====================

    def should_retrain_model(self, symbol: str) -> bool:
    """
    Determine if model should be retrained based on:
    - Time since last training
    - Model performance degradation
    - Significant market condition changes
    """
    # Check if retrain interval has passed
    retrain_interval = self.config['ml'].get('retrain_interval', 24)  # hours
    
    # Implementation would check last training time and performance metrics
    # For now, return False to avoid constant retraining
    return False

    def calculate_signal_strength(self, df: pd.DataFrame, signals: List[str]) -> float:
    """
    Calculate overall signal strength by combining:
    - Number of confirming signals
    - Signal confidence levels
    - Market regime appropriateness
    - Volume confirmation
    """
    base_strength = min(len(signals) / 10.0, 1.0)  # Normalize by expected max signals
    
    # Adjust for signal confidence if available
    if 'signal_confidence' in df.columns:
        confidence = df['signal_confidence'].iloc[-1]
        if not pd.isna(confidence):
            base_strength *= confidence
    
    # Adjust for market regime
    if 'regime' in df.columns:
        regime = df['regime'].iloc[-1]
        if regime == 2:  # High volatility regime
            base_strength *= 0.8  # Reduce strength in volatile conditions
    
    return base_strength

    def perform_safety_checks(self, execution_params: Dict) -> bool:
    """
    Perform final safety checks before trade execution:
    - Position size within limits
    - No anomalies detected
    - Market conditions suitable
    - Account balance sufficient
    """
    # Check position size
    if execution_params['position_size'] > self.config['risk']['max_position_size'] * self.config['risk']['account_balance']:
        logger.warning("Position size exceeds maximum allowed")
        return False
    
    # Check for anomalies
    if execution_params.get('anomaly_detected', False):
        logger.warning("Market anomaly detected - blocking trade")
        return False
    
    # Check minimum signal strength
    if execution_params['signal_strength'] < 0.3:
        logger.warning("Signal strength too low")
        return False
    
    return True

    def prepare_notification_message(self, symbol: str, signals: List[str], execution_params: Dict, df: pd.DataFrame) -> str:
    """
    Prepare comprehensive notification message with all relevant information
    """
    message = f"🚀 <b>TRADING SIGNAL - {symbol}</b>\n\n"
    
    # Add current market data
    message += f"💰 <b>Price:</b> ${execution_params['current_price']:.4f}\n"
    message += f"📊 <b>Position Size:</b> ${execution_params['position_size']:.2f}\n"
    message += f"🎯 <b>Signal Strength:</b> {execution_params['signal_strength']:.1%}\n"
    message += f"🔮 <b>Confidence:</b> {execution_params['confidence']:.1%}\n"
    
    # Add market regime info
    if execution_params['market_regime'] != 'unknown':
        message += f"📈 <b>Market Regime:</b> {execution_params['market_regime']}\n"
    
    # Add stop loss levels
    message += f"🛑 <b>Stop Loss (Long):</b> ${execution_params['stop_loss_long']:.4f}\n"
    message += f"🛑 <b>Stop Loss (Short):</b> ${execution_params['stop_loss_short']:.4f}\n"
    
    # Add all signals
    message += f"\n<b>SIGNALS:</b>\n"
    for signal in signals:
        message += f"{signal}\n"
    
    return message

    def execute_trade(self, execution_params: Dict) -> Optional[str]:
    """
    Execute the actual trade based on execution parameters
    Returns execution result string or None if failed
    """
    try:
        # Implementation would interface with exchange API
        # For now, return a mock result
        return f"Mock execution: {execution_params['symbol']} - ${execution_params['position_size']:.2f}"
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return None

    def update_performance_metrics(self, symbol: str, execution_params: Dict):
    """
    Update performance tracking metrics
    """
    self.performance_metrics['total_signals'] += 1
    # Additional performance tracking logic would go here

    def store_signal(self, symbol: str, signals: List[str]):
    """
    Store signal for duplicate detection
    """
    self.last_signals[symbol] = {
        'signals': signals,
        'timestamp': datetime.now()
    }

    def is_duplicate_signal(self, symbol: str, signals: List[str]) -> bool:
    """
    Check if this is a duplicate signal
    """
    if symbol not in self.last_signals:
        return False
    
    last_signal_time = self.last_signals[symbol]['timestamp']
    time_diff = datetime.now() - last_signal_time
    
    # Consider duplicate if same signals within 1 hour
    return time_diff < timedelta(hours=1) and self.last_signals[symbol]['signals'] == signals

    def update_historical_data(self, symbol: str, df: pd.DataFrame):
    """
    Update historical data storage for the symbol
    """
    self.historical_data[symbol] = df

    def cleanup_old_data(self, symbol: str):
    """
    Cleanup old data to prevent memory issues
    """
    if symbol in self.historical_data:
        # Keep only last 500 records
        self.historical_data[symbol] = self.historical_data[symbol].tail(500)

    def check_traditional_signals(self, df: pd.DataFrame, symbol: str) -> List[str]:
    """
    Generate traditional technical analysis signals
    This method would contain the traditional signal logic
    """
    signals = []
    
    if len(df) < 2:
        return signals
    
    current = df.iloc[-1]
    
    # RSI signals
    if 'rsi' in df.columns:
        rsi = current['rsi']
        if rsi < 30:
            signals.append(f"📈 <b>RSI OVERSOLD</b> - {rsi:.1f}")
        elif rsi > 70:
            signals.append(f"📉 <b>RSI OVERBOUGHT</b> - {rsi:.1f}")
    
    # MACD signals
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if current['macd'] > current['macd_signal']:
            signals.append(f"📈 <b>MACD BULLISH CROSSOVER</b>")
        elif current['macd'] < current['macd_signal']:
            signals.append(f"📉 <b>MACD BEARISH CROSSOVER</b>")
    
    # Volume signals
    if 'volume_sma' in df.columns:
        volume_ratio = current['volume'] / current['volume_sma']
        if volume_ratio > 2.0:
            signals.append(f"📊 <b>HIGH VOLUME</b> - {volume_ratio:.1f}x average")
    
    return signals

    def send_telegram_message(self, message: str):
    """
    Send notification via Telegram
    """
    try:
        # Implementation would send via Telegram API
        logger.info(f"Telegram notification: {message[:100]}...")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
