"""
Paper trading module.
Simulates live trading with virtual money.
"""

from .simulator import PaperTrader, Position, PaperTradeSession

__all__ = ['PaperTrader', 'Position', 'PaperTradeSession']
