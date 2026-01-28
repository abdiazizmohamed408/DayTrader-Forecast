"""
Prediction Tracker Module.
Logs predictions and verifies outcomes by checking if price hit target or stop-loss.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .database import PredictionDatabase
from data.fetcher import DataFetcher


class PredictionTracker:
    """
    Tracks predictions and verifies their outcomes.
    
    When a signal is generated, it's logged to the database.
    Later, we verify if the price hit the target or stop-loss.
    """
    
    def __init__(self, db_path: str = "predictions.db", max_days: int = 10):
        """
        Initialize the tracker.
        
        Args:
            db_path: Path to SQLite database
            max_days: Maximum days to track a prediction before expiring
        """
        self.db = PredictionDatabase(db_path)
        self.fetcher = DataFetcher()
        self.max_days = max_days
    
    def log_prediction(
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
        Log a new prediction to the database.
        
        Args:
            ticker: Stock symbol
            signal_type: BUY, SELL, or HOLD
            entry_price: Current price at signal time
            target_price: Target price for the trade
            stop_loss: Stop loss price
            probability: Confidence score (0-100)
            sentiment: BULLISH, BEARISH, or NEUTRAL
            indicator_scores: Scores from each indicator
            indicator_values: Raw indicator values
            reasons: Signal reasoning
            
        Returns:
            Database ID of the logged prediction
        """
        # Only track actionable signals (BUY or SELL)
        if signal_type == "HOLD":
            return -1
        
        return self.db.add_prediction(
            ticker=ticker,
            signal_type=signal_type,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            probability=probability,
            sentiment=sentiment,
            indicator_scores=indicator_scores,
            indicator_values=indicator_values,
            reasons=reasons
        )
    
    def verify_outcomes(self, verbose: bool = False) -> Dict:
        """
        Check all pending predictions and update their outcomes.
        
        Fetches historical data and checks if:
        - Price hit target (WIN)
        - Price hit stop-loss (LOSS)
        - Neither within max_days (EXPIRED)
        
        Args:
            verbose: Print progress information
            
        Returns:
            Summary of verification results
        """
        pending = self.db.get_pending_predictions(max_age_days=self.max_days + 5)
        
        results = {
            'checked': 0,
            'wins': 0,
            'losses': 0,
            'expired': 0,
            'still_pending': 0,
            'errors': 0
        }
        
        for pred in pending:
            results['checked'] += 1
            
            try:
                outcome = self._check_single_prediction(pred, verbose)
                
                if outcome == 'WIN':
                    results['wins'] += 1
                elif outcome == 'LOSS':
                    results['losses'] += 1
                elif outcome == 'EXPIRED':
                    results['expired'] += 1
                else:
                    results['still_pending'] += 1
                    
            except Exception as e:
                results['errors'] += 1
                if verbose:
                    print(f"  ❌ Error checking {pred['ticker']}: {e}")
        
        return results
    
    def _check_single_prediction(
        self, pred: Dict, verbose: bool = False
    ) -> Optional[str]:
        """
        Check a single prediction's outcome.
        
        Args:
            pred: Prediction dictionary from database
            verbose: Print progress
            
        Returns:
            Outcome string or None if still pending
        """
        ticker = pred['ticker']
        signal_type = pred['signal_type']
        entry_price = pred['entry_price']
        target_price = pred['target_price']
        stop_loss = pred['stop_loss']
        created_at = datetime.fromisoformat(pred['created_at'])
        
        if verbose:
            print(f"  Checking {ticker} ({signal_type} @ ${entry_price:.2f})...", end=" ")
        
        # Get historical data since prediction
        data = self.fetcher.get_stock_data(ticker, period="1mo", interval="1d")
        if data is None or data.empty:
            if verbose:
                print("No data")
            return None
        
        # Filter to dates after prediction (handle timezone-aware index)
        try:
            # Convert created_at to timezone-aware if index is timezone-aware
            if data.index.tz is not None:
                from datetime import timezone
                created_at_tz = created_at.replace(tzinfo=timezone.utc)
                # Convert to same timezone as data
                created_at_tz = created_at_tz.astimezone(data.index.tz)
                data = data[data.index >= created_at_tz]
            else:
                data = data[data.index >= created_at]
        except Exception:
            # Fallback: just use all recent data
            pass
        
        if data.empty:
            if verbose:
                print("No new data yet")
            return None
        
        days_elapsed = (datetime.now() - created_at).days
        
        # Check each day's price action
        for i, (date, row) in enumerate(data.iterrows()):
            high = row['high']
            low = row['low']
            close = row['close']
            days_held = i + 1
            
            if signal_type == 'BUY':
                # For BUY: target hit = WIN, stop hit = LOSS
                if target_price and high >= target_price:
                    profit_pct = ((target_price - entry_price) / entry_price) * 100
                    self.db.update_outcome(
                        pred['id'], 'WIN', target_price, profit_pct, days_held
                    )
                    if verbose:
                        print(f"✅ WIN (+{profit_pct:.1f}%)")
                    return 'WIN'
                
                if stop_loss and low <= stop_loss:
                    loss_pct = ((stop_loss - entry_price) / entry_price) * 100
                    self.db.update_outcome(
                        pred['id'], 'LOSS', stop_loss, loss_pct, days_held
                    )
                    if verbose:
                        print(f"❌ LOSS ({loss_pct:.1f}%)")
                    return 'LOSS'
                    
            elif signal_type == 'SELL':
                # For SELL: target hit (lower) = WIN, stop hit (higher) = LOSS
                if target_price and low <= target_price:
                    profit_pct = ((entry_price - target_price) / entry_price) * 100
                    self.db.update_outcome(
                        pred['id'], 'WIN', target_price, profit_pct, days_held
                    )
                    if verbose:
                        print(f"✅ WIN (+{profit_pct:.1f}%)")
                    return 'WIN'
                
                if stop_loss and high >= stop_loss:
                    loss_pct = ((entry_price - stop_loss) / entry_price) * 100
                    self.db.update_outcome(
                        pred['id'], 'LOSS', stop_loss, loss_pct, days_held
                    )
                    if verbose:
                        print(f"❌ LOSS ({loss_pct:.1f}%)")
                    return 'LOSS'
        
        # Check if expired
        if days_elapsed >= self.max_days:
            # Mark as expired with current P/L
            current_close = data['close'].iloc[-1]
            
            if signal_type == 'BUY':
                pl_pct = ((current_close - entry_price) / entry_price) * 100
            else:
                pl_pct = ((entry_price - current_close) / entry_price) * 100
            
            outcome = 'WIN' if pl_pct > 0 else 'LOSS'
            self.db.update_outcome(
                pred['id'], outcome, current_close, pl_pct, days_elapsed
            )
            if verbose:
                emoji = "✅" if outcome == 'WIN' else "❌"
                print(f"{emoji} EXPIRED as {outcome} ({pl_pct:+.1f}%)")
            return outcome
        
        if verbose:
            print("⏳ Still pending")
        return None
    
    def get_recent_predictions(self, limit: int = 20) -> List[Dict]:
        """Get most recent predictions."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT * FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        return self.db.get_stats()
    
    def get_ticker_stats(self) -> List[Dict]:
        """Get statistics by ticker."""
        return self.db.get_stats_by_ticker()
    
    def get_indicator_stats(self) -> Dict[str, Dict]:
        """Get statistics by indicator."""
        return self.db.get_stats_by_indicator()
    
    def close(self):
        """Close database connection."""
        self.db.close()
