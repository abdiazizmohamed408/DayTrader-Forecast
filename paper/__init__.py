"""
Paper trading module.
Simulates live trading with virtual money.
"""

from .simulator import PaperTrader, Position, PaperTradeSession, ClosedPosition

__all__ = ['PaperTrader', 'Position', 'PaperTradeSession', 'ClosedPosition']
