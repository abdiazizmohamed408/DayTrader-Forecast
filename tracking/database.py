"""
SQLite Database Module for Prediction Tracking.
Stores all predictions and their outcomes for backtesting and analysis.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class PredictionDatabase:
    """
    SQLite database handler for storing and retrieving predictions.
    
    Stores prediction data including entry prices, targets, stop-losses,
    and tracks outcomes (WIN/LOSS/PENDING).
    """
    
    def __init__(self, db_path: str = "predictions.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                probability REAL NOT NULL,
                sentiment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Outcome tracking
                outcome TEXT DEFAULT 'PENDING',
                exit_price REAL,
                exit_date TIMESTAMP,
                profit_loss_pct REAL,
                days_held INTEGER,
                
                -- Indicator scores at time of prediction
                rsi_score REAL,
                macd_score REAL,
                ma_score REAL,
                bb_score REAL,
                volume_score REAL,
                sr_score REAL,
                
                -- Raw indicator values
                rsi_value REAL,
                macd_value REAL,
                reasons TEXT
            )
        """)
        
        # Performance summary table (cached for speed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_cache (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                indicator TEXT,
                timeframe TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                pending INTEGER,
                win_rate REAL,
                avg_profit REAL,
                avg_loss REAL,
                profit_factor REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Weight adjustment history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weight_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator TEXT NOT NULL,
                old_weight REAL,
                new_weight REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_ticker 
            ON predictions(ticker)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_outcome 
            ON predictions(outcome)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_date 
            ON predictions(created_at)
        """)
        
        self.conn.commit()
    
    def add_prediction(
        self,
        ticker: str,
        signal_type: str,
        entry_price: float,
        target_price: Optional[float],
        stop_loss: Optional[float],
        probability: float,
        sentiment: str,
        indicator_scores: Optional[Dict] = None,
        indicator_values: Optional[Dict] = None,
        reasons: Optional[List[str]] = None
    ) -> int:
        """
        Add a new prediction to the database.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY, SELL, or HOLD
            entry_price: Price at signal generation
            target_price: Target price
            stop_loss: Stop loss price
            probability: Confidence score
            sentiment: BULLISH, BEARISH, or NEUTRAL
            indicator_scores: Dict of indicator bullish/bearish scores
            indicator_values: Dict of raw indicator values
            reasons: List of signal reasons
            
        Returns:
            ID of the inserted prediction
        """
        cursor = self.conn.cursor()
        
        # Extract indicator scores
        rsi_score = indicator_scores.get('rsi', {}).get('bullish', 0) if indicator_scores else None
        macd_score = indicator_scores.get('macd', {}).get('bullish', 0) if indicator_scores else None
        ma_score = indicator_scores.get('moving_averages', {}).get('bullish', 0) if indicator_scores else None
        bb_score = indicator_scores.get('bollinger_bands', {}).get('bullish', 0) if indicator_scores else None
        volume_score = indicator_scores.get('volume', {}).get('bullish', 0) if indicator_scores else None
        sr_score = indicator_scores.get('support_resistance', {}).get('bullish', 0) if indicator_scores else None
        
        # Extract raw values
        rsi_value = indicator_values.get('rsi') if indicator_values else None
        macd_value = indicator_values.get('macd') if indicator_values else None
        
        cursor.execute("""
            INSERT INTO predictions (
                ticker, signal_type, entry_price, target_price, stop_loss,
                probability, sentiment, rsi_score, macd_score, ma_score,
                bb_score, volume_score, sr_score, rsi_value, macd_value, reasons
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, signal_type, entry_price, target_price, stop_loss,
            probability, sentiment, rsi_score, macd_score, ma_score,
            bb_score, volume_score, sr_score, rsi_value, macd_value,
            json.dumps(reasons) if reasons else None
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def update_outcome(
        self,
        prediction_id: int,
        outcome: str,
        exit_price: float,
        profit_loss_pct: float,
        days_held: int
    ):
        """
        Update the outcome of a prediction.
        
        Args:
            prediction_id: ID of the prediction
            outcome: WIN, LOSS, or EXPIRED
            exit_price: Price at exit
            profit_loss_pct: Percentage profit/loss
            days_held: Number of days position was held
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE predictions
            SET outcome = ?, exit_price = ?, exit_date = ?,
                profit_loss_pct = ?, days_held = ?
            WHERE id = ?
        """, (outcome, exit_price, datetime.now(), profit_loss_pct, days_held, prediction_id))
        self.conn.commit()
    
    def get_pending_predictions(self, max_age_days: int = 30) -> List[Dict]:
        """
        Get all pending predictions that need outcome verification.
        
        Args:
            max_age_days: Maximum age of predictions to check
            
        Returns:
            List of pending prediction dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE outcome = 'PENDING'
            AND created_at >= datetime('now', ?)
            ORDER BY created_at DESC
        """, (f'-{max_age_days} days',))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_predictions_by_ticker(self, ticker: str, limit: int = 100) -> List[Dict]:
        """Get predictions for a specific ticker."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE ticker = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (ticker, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_completed(self, limit: int = 1000) -> List[Dict]:
        """Get all completed predictions (WIN or LOSS)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            WHERE outcome IN ('WIN', 'LOSS')
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get overall prediction statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending,
                AVG(CASE WHEN outcome = 'WIN' THEN profit_loss_pct END) as avg_win,
                AVG(CASE WHEN outcome = 'LOSS' THEN profit_loss_pct END) as avg_loss,
                SUM(CASE WHEN outcome = 'WIN' THEN profit_loss_pct ELSE 0 END) as total_profit,
                SUM(CASE WHEN outcome = 'LOSS' THEN ABS(profit_loss_pct) ELSE 0 END) as total_loss
            FROM predictions
        """)
        
        row = cursor.fetchone()
        
        total_profit = row['total_profit'] or 0
        total_loss = row['total_loss'] or 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        wins = row['wins'] or 0
        losses = row['losses'] or 0
        completed = wins + losses
        win_rate = (wins / completed * 100) if completed > 0 else 0
        
        return {
            'total_predictions': row['total'] or 0,
            'wins': wins,
            'losses': losses,
            'pending': row['pending'] or 0,
            'win_rate': win_rate,
            'avg_win_pct': row['avg_win'] or 0,
            'avg_loss_pct': row['avg_loss'] or 0,
            'profit_factor': profit_factor,
            'total_profit_pct': total_profit,
            'total_loss_pct': total_loss
        }
    
    def get_stats_by_ticker(self) -> List[Dict]:
        """Get statistics grouped by ticker."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                ticker,
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN outcome = 'WIN' THEN profit_loss_pct END) as avg_win,
                AVG(CASE WHEN outcome = 'LOSS' THEN profit_loss_pct END) as avg_loss
            FROM predictions
            WHERE outcome IN ('WIN', 'LOSS')
            GROUP BY ticker
            ORDER BY wins DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            wins = row['wins'] or 0
            losses = row['losses'] or 0
            total = wins + losses
            results.append({
                'ticker': row['ticker'],
                'total': total,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'avg_win': row['avg_win'] or 0,
                'avg_loss': row['avg_loss'] or 0
            })
        
        return results
    
    def get_stats_by_indicator(self) -> Dict[str, Dict]:
        """
        Analyze which indicators contribute most to winning trades.
        
        Returns:
            Dictionary of indicator performance stats
        """
        cursor = self.conn.cursor()
        
        indicators = {
            'rsi': 'rsi_score',
            'macd': 'macd_score',
            'moving_averages': 'ma_score',
            'bollinger_bands': 'bb_score',
            'volume': 'volume_score',
            'support_resistance': 'sr_score'
        }
        
        results = {}
        
        for name, column in indicators.items():
            # Get correlation between indicator score and win rate
            cursor.execute(f"""
                SELECT 
                    AVG(CASE WHEN outcome = 'WIN' THEN {column} END) as avg_win_score,
                    AVG(CASE WHEN outcome = 'LOSS' THEN {column} END) as avg_loss_score,
                    COUNT(CASE WHEN outcome = 'WIN' AND {column} > 0.5 THEN 1 END) as high_score_wins,
                    COUNT(CASE WHEN outcome = 'LOSS' AND {column} > 0.5 THEN 1 END) as high_score_losses,
                    COUNT(CASE WHEN outcome = 'WIN' AND {column} <= 0.5 THEN 1 END) as low_score_wins,
                    COUNT(CASE WHEN outcome = 'LOSS' AND {column} <= 0.5 THEN 1 END) as low_score_losses
                FROM predictions
                WHERE outcome IN ('WIN', 'LOSS')
                AND {column} IS NOT NULL
            """)
            
            row = cursor.fetchone()
            
            high_total = (row['high_score_wins'] or 0) + (row['high_score_losses'] or 0)
            high_win_rate = (row['high_score_wins'] / high_total * 100) if high_total > 0 else 0
            
            low_total = (row['low_score_wins'] or 0) + (row['low_score_losses'] or 0)
            low_win_rate = (row['low_score_wins'] / low_total * 100) if low_total > 0 else 0
            
            # Effectiveness = how much better high scores perform vs low scores
            effectiveness = high_win_rate - low_win_rate
            
            results[name] = {
                'avg_win_score': row['avg_win_score'] or 0,
                'avg_loss_score': row['avg_loss_score'] or 0,
                'high_score_win_rate': high_win_rate,
                'low_score_win_rate': low_win_rate,
                'effectiveness': effectiveness,
                'high_score_trades': high_total,
                'low_score_trades': low_total
            }
        
        return results
    
    def save_weight_adjustment(
        self,
        indicator: str,
        old_weight: float,
        new_weight: float,
        reason: str
    ):
        """Save a weight adjustment for history tracking."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO weight_history (indicator, old_weight, new_weight, reason)
            VALUES (?, ?, ?, ?)
        """, (indicator, old_weight, new_weight, reason))
        self.conn.commit()
    
    def get_weight_history(self, limit: int = 50) -> List[Dict]:
        """Get weight adjustment history."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM weight_history
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
