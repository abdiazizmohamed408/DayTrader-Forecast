"""
Performance Tracking Module.
Logs every signal to SQLite and tracks outcomes for accuracy measurement.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class PredictionTracker:
    """
    Tracks trading signal predictions and their outcomes.
    
    Stores all predictions in SQLite and evaluates whether
    price hit target (WIN) or stop-loss (LOSS).
    """
    
    def __init__(self, db_path: str = "data/predictions.db"):
        """
        Initialize tracker with database path.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                probability REAL NOT NULL,
                outcome TEXT DEFAULT 'PENDING',
                outcome_price REAL,
                outcome_timestamp TEXT,
                profit_pct REAL,
                market_context TEXT,
                volume_confirmed INTEGER DEFAULT 0,
                timeframe_alignment REAL,
                reasons TEXT
            )
        ''')
        
        # Index for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
            ON predictions(ticker, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_outcome 
            ON predictions(outcome)
        ''')
        
        conn.commit()
        conn.close()
    
    def log_signal(
        self,
        ticker: str,
        signal_type: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        probability: float,
        market_context: Optional[Dict] = None,
        volume_confirmed: bool = False,
        timeframe_alignment: Optional[float] = None,
        reasons: Optional[List[str]] = None
    ) -> int:
        """
        Log a new trading signal prediction.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY or SELL
            entry_price: Entry price
            target_price: Target price
            stop_loss: Stop loss price
            probability: Signal probability (0-100)
            market_context: Optional market context dict
            volume_confirmed: Whether volume confirmation passed
            timeframe_alignment: Multi-timeframe alignment score
            reasons: List of signal reasons
            
        Returns:
            ID of the inserted prediction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, ticker, signal_type, entry_price, target_price, 
             stop_loss, probability, market_context, volume_confirmed,
             timeframe_alignment, reasons)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            ticker.upper(),
            signal_type,
            entry_price,
            target_price,
            stop_loss,
            probability,
            json.dumps(market_context) if market_context else None,
            1 if volume_confirmed else 0,
            timeframe_alignment,
            json.dumps(reasons) if reasons else None
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def update_pending_outcomes(self, fetcher) -> Dict[str, int]:
        """
        Check and update all pending predictions.
        
        Args:
            fetcher: DataFetcher instance for getting current prices
            
        Returns:
            Dict with counts of wins, losses, still_pending
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, ticker, timestamp, signal_type, entry_price, 
                   target_price, stop_loss FROM predictions 
            WHERE outcome = 'PENDING'
        ''')
        
        pending = cursor.fetchall()
        conn.close()
        
        results = {'wins': 0, 'losses': 0, 'still_pending': 0}
        
        for row in pending:
            pred_id, ticker, timestamp, signal_type, entry_price, target_price, stop_loss = row
            
            try:
                # Get price data since signal
                data = fetcher.get_stock_data(ticker, period='5d', interval='1h')
                if data is not None and len(data) > 0:
                    signal_time = datetime.fromisoformat(timestamp)
                    
                    # Filter data to after signal time
                    if hasattr(data.index, 'tz_localize'):
                        try:
                            recent_data = data[data.index >= signal_time]
                        except:
                            recent_data = data  # Use all data if filtering fails
                    else:
                        recent_data = data
                    
                    if len(recent_data) > 0:
                        high_since = recent_data['high'].max()
                        low_since = recent_data['low'].min()
                        current = recent_data['close'].iloc[-1]
                        
                        outcome = self._check_single_outcome(
                            signal_type, entry_price, target_price, stop_loss,
                            current, high_since, low_since
                        )
                        
                        if outcome:
                            # Update the prediction
                            self._update_outcome(
                                pred_id, outcome, 
                                target_price if outcome == 'WIN' else stop_loss,
                                entry_price, signal_type
                            )
                            
                            if outcome == 'WIN':
                                results['wins'] += 1
                            else:
                                results['losses'] += 1
                        else:
                            results['still_pending'] += 1
                    else:
                        results['still_pending'] += 1
                else:
                    results['still_pending'] += 1
            except Exception as e:
                results['still_pending'] += 1
        
        return results
    
    def _check_single_outcome(
        self,
        signal_type: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        current_price: float,
        high_price: float,
        low_price: float
    ) -> Optional[str]:
        """Check if a single prediction hit target or stop."""
        if signal_type == 'BUY':
            if high_price >= target_price:
                return 'WIN'
            elif low_price <= stop_loss:
                return 'LOSS'
        elif signal_type == 'SELL':
            if low_price <= target_price:
                return 'WIN'
            elif high_price >= stop_loss:
                return 'LOSS'
        return None
    
    def _update_outcome(
        self,
        pred_id: int,
        outcome: str,
        exit_price: float,
        entry_price: float,
        signal_type: str
    ):
        """Update a prediction's outcome in the database."""
        if signal_type == 'BUY':
            profit_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            profit_pct = ((entry_price - exit_price) / entry_price) * 100
        
        if outcome == 'LOSS':
            profit_pct = -abs(profit_pct)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET outcome = ?, outcome_price = ?, outcome_timestamp = ?, profit_pct = ?
            WHERE id = ?
        ''', (outcome, exit_price, datetime.now().isoformat(), profit_pct, pred_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(
        self,
        ticker: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict:
        """
        Calculate performance statistics.
        
        Args:
            ticker: Filter by ticker (optional)
            days: Filter to last N days (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM predictions WHERE outcome != 'PENDING'"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        
        if days:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            query += " AND timestamp >= ?"
            params.append(cutoff)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'total_predictions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_return': 0
            }
        
        predictions = [dict(row) for row in rows]
        
        wins = [p for p in predictions if p['outcome'] == 'WIN']
        losses = [p for p in predictions if p['outcome'] == 'LOSS']
        
        total = len(predictions)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = (win_count / total * 100) if total > 0 else 0
        
        # Average profit/loss percentages
        avg_profit = sum(abs(p['profit_pct']) for p in wins) / win_count if wins else 0
        avg_loss = sum(abs(p['profit_pct']) for p in losses) / loss_count if losses else 0
        
        # Profit factor = gross profits / gross losses
        gross_profit = sum(abs(p['profit_pct']) for p in wins)
        gross_loss = sum(abs(p['profit_pct']) for p in losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit if gross_profit > 0 else 0
        
        # Total return (sum of all profit percentages)
        total_return = sum(p['profit_pct'] for p in predictions)
        
        return {
            'total_predictions': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return
        }
    
    def get_ticker_performance(self) -> List[Dict]:
        """
        Get performance breakdown by ticker.
        
        Returns:
            List of dicts with per-ticker statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT ticker FROM predictions WHERE outcome != 'PENDING'
        ''')
        
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        results = []
        for ticker in tickers:
            stats = self.get_performance_stats(ticker=ticker)
            stats['ticker'] = ticker
            results.append(stats)
        
        # Sort by win rate
        results.sort(key=lambda x: x['win_rate'], reverse=True)
        return results
    
    def get_pending_predictions(self) -> List[Dict]:
        """Get all pending predictions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE outcome = 'PENDING'
            ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """
        Get recent predictions with their outcomes.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of prediction dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
