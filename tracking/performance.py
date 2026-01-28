"""
Performance Analysis Module.
Analyzes historical prediction accuracy and generates performance reports.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .database import PredictionDatabase


class PerformanceAnalyzer:
    """
    Analyzes prediction performance and calculates optimal weights.
    
    Uses historical win/loss data to determine which indicators
    are most predictive and adjusts weights accordingly.
    """
    
    def __init__(self, db_path: str = "predictions.db"):
        """
        Initialize the analyzer.
        
        Args:
            db_path: Path to predictions database
        """
        self.db = PredictionDatabase(db_path)
    
    def get_summary(self) -> Dict:
        """
        Get complete performance summary.
        
        Returns:
            Dictionary with all performance metrics
        """
        stats = self.db.get_stats()
        ticker_stats = self.db.get_stats_by_ticker()
        indicator_stats = self.db.get_stats_by_indicator()
        
        return {
            'overall': stats,
            'by_ticker': ticker_stats,
            'by_indicator': indicator_stats
        }
    
    def calculate_optimal_weights(
        self,
        current_weights: Dict[str, float],
        min_trades: int = 10,
        learning_rate: float = 0.1
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Calculate optimal indicator weights based on historical performance.
        
        Uses indicator effectiveness (difference in win rate between
        high-score and low-score trades) to adjust weights.
        
        Args:
            current_weights: Current indicator weights
            min_trades: Minimum trades needed to adjust an indicator
            learning_rate: How aggressively to adjust weights (0-1)
            
        Returns:
            Tuple of (new_weights, adjustment_reasons)
        """
        indicator_stats = self.db.get_stats_by_indicator()
        
        new_weights = current_weights.copy()
        reasons = []
        
        # Calculate effectiveness scores
        effectiveness = {}
        for indicator, stats in indicator_stats.items():
            total_trades = stats['high_score_trades'] + stats['low_score_trades']
            
            if total_trades >= min_trades:
                # Effectiveness = how much better high scores perform
                effectiveness[indicator] = stats['effectiveness']
            else:
                effectiveness[indicator] = 0  # Not enough data
        
        if not any(effectiveness.values()):
            return new_weights, ["Not enough data to optimize weights"]
        
        # Normalize effectiveness to use as weight adjustments
        max_eff = max(abs(e) for e in effectiveness.values() if e != 0)
        if max_eff == 0:
            return new_weights, ["All indicators performing similarly"]
        
        for indicator, eff in effectiveness.items():
            if indicator not in current_weights:
                continue
            
            old_weight = current_weights[indicator]
            
            # Scale adjustment by learning rate
            adjustment = (eff / max_eff) * learning_rate * 0.1
            
            # Apply adjustment
            new_weight = old_weight + adjustment
            
            # Clamp to reasonable range [0.05, 0.4]
            new_weight = max(0.05, min(0.4, new_weight))
            
            if abs(new_weight - old_weight) > 0.005:
                new_weights[indicator] = new_weight
                direction = "increased" if new_weight > old_weight else "decreased"
                reasons.append(
                    f"{indicator}: {old_weight:.2f} → {new_weight:.2f} "
                    f"({direction}, effectiveness: {eff:+.1f}%)"
                )
                
                # Log adjustment
                self.db.save_weight_adjustment(
                    indicator, old_weight, new_weight,
                    f"Effectiveness: {eff:+.1f}%"
                )
        
        # Normalize weights to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        
        if not reasons:
            reasons.append("No significant weight adjustments needed")
        
        return new_weights, reasons
    
    def generate_report(self) -> str:
        """
        Generate a detailed performance report.
        
        Returns:
            Markdown formatted report string
        """
        stats = self.db.get_stats()
        ticker_stats = self.db.get_stats_by_ticker()
        indicator_stats = self.db.get_stats_by_indicator()
        
        report = f"""# 📊 Prediction Performance Report

**Generated:** {datetime.now().strftime("%B %d, %Y %H:%M")}

---

## 📈 Overall Performance

| Metric | Value |
|--------|-------|
| **Total Predictions** | {stats['total_predictions']} |
| **Wins** | {stats['wins']} |
| **Losses** | {stats['losses']} |
| **Pending** | {stats['pending']} |
| **Win Rate** | **{stats['win_rate']:.1f}%** |
| **Avg Win** | +{stats['avg_win_pct']:.2f}% |
| **Avg Loss** | {stats['avg_loss_pct']:.2f}% |
| **Profit Factor** | {stats['profit_factor']:.2f} |

"""
        
        # Win rate visualization
        if stats['wins'] + stats['losses'] > 0:
            win_pct = stats['win_rate']
            bar_len = 20
            filled = int(win_pct / 100 * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            report += f"""### Win Rate Visualization

```
[{bar}] {win_pct:.1f}%
 {stats['wins']} wins / {stats['losses']} losses
```

"""
        
        # Performance by ticker
        if ticker_stats:
            report += """## 📊 Performance by Ticker

| Ticker | Trades | Wins | Losses | Win Rate | Avg Win | Avg Loss |
|--------|--------|------|--------|----------|---------|----------|
"""
            for t in sorted(ticker_stats, key=lambda x: x['win_rate'], reverse=True):
                report += (
                    f"| {t['ticker']} | {t['total']} | {t['wins']} | {t['losses']} | "
                    f"{t['win_rate']:.1f}% | +{t['avg_win']:.1f}% | {t['avg_loss']:.1f}% |\n"
                )
            report += "\n"
        
        # Performance by indicator
        if indicator_stats:
            report += """## 🔬 Indicator Effectiveness

This shows how well each indicator predicts successful trades.
**Effectiveness** = Win rate when indicator score > 0.5 minus win rate when score ≤ 0.5

| Indicator | High Score WR | Low Score WR | Effectiveness | Trades |
|-----------|---------------|--------------|---------------|--------|
"""
            sorted_indicators = sorted(
                indicator_stats.items(),
                key=lambda x: x[1]['effectiveness'],
                reverse=True
            )
            
            for name, stats in sorted_indicators:
                total_trades = stats['high_score_trades'] + stats['low_score_trades']
                eff_emoji = "🟢" if stats['effectiveness'] > 5 else (
                    "🔴" if stats['effectiveness'] < -5 else "🟡"
                )
                report += (
                    f"| {name} | {stats['high_score_win_rate']:.1f}% | "
                    f"{stats['low_score_win_rate']:.1f}% | "
                    f"{eff_emoji} {stats['effectiveness']:+.1f}% | {total_trades} |\n"
                )
            
            report += """
### Interpretation

- 🟢 **Positive effectiveness** = Indicator is predictive (high scores = more wins)
- 🔴 **Negative effectiveness** = Indicator may be inversely predictive
- 🟡 **Near zero** = Indicator has little predictive value

"""
        
        # Recommendations
        report += """## 💡 Recommendations

"""
        if stats['total_predictions'] < 20:
            report += "- ⚠️ **More data needed**: At least 20 completed trades recommended for reliable statistics\n"
        
        if stats['win_rate'] >= 55:
            report += "- ✅ **Good performance**: Win rate above 55% indicates a positive edge\n"
        elif stats['win_rate'] < 45:
            report += "- ⚠️ **Review strategy**: Win rate below 45% - consider adjusting parameters\n"
        
        if stats['profit_factor'] >= 1.5:
            report += "- ✅ **Strong profit factor**: Profits exceed losses by 50%+\n"
        elif stats['profit_factor'] < 1.0:
            report += "- ⚠️ **Negative profit factor**: Losses exceed profits - review risk management\n"
        
        # Best/worst indicators
        if indicator_stats:
            sorted_ind = sorted(
                indicator_stats.items(),
                key=lambda x: x[1]['effectiveness'],
                reverse=True
            )
            best = sorted_ind[0]
            worst = sorted_ind[-1]
            
            if best[1]['effectiveness'] > 10:
                report += f"- 📈 **Best indicator**: {best[0]} ({best[1]['effectiveness']:+.1f}% effectiveness)\n"
            if worst[1]['effectiveness'] < -10:
                report += f"- 📉 **Weakest indicator**: {worst[0]} ({worst[1]['effectiveness']:+.1f}% effectiveness)\n"
        
        report += """
---

*Past performance does not guarantee future results. Always use proper risk management.*
"""
        
        return report
    
    def get_weight_history(self) -> List[Dict]:
        """Get history of weight adjustments."""
        return self.db.get_weight_history()
    
    def close(self):
        """Close database connection."""
        self.db.close()
