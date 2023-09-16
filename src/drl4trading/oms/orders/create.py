from tensortrade.oms.instruments.quantity import Quantity
from tensortrade.oms.orders.criteria import Stop
from tensortrade.oms.orders.trade import TradeSide, TradeType
from tensortrade.oms.orders.order import Order
from tensortrade.oms.orders.order_spec import OrderSpec
from tensortrade.oms.instruments.exchange_pair import ExchangePair
from tensortrade.oms.wallets.portfolio import Portfolio
from src.drl4trading.oms.orders.criteria import DailyTL

def risk_managed_dtl_order(side: "TradeSide",
                       trade_type: "TradeType",
                       exchange_pair: "ExchangePair",
                       price: float,
                       quantity: "Quantity",
                       down_percent: float,
                       up_percent: float,
                       portfolio: "Portfolio",
                       start: int = None,
                       end: int = None):
    """Create a stop order that manages for percentages above and below the
    entry price of the order.

    Parameters
    ----------
    side : `TradeSide`
        The side of the order.
    trade_type : `TradeType`
        The type of trade to make when going in.
    exchange_pair : `ExchangePair`
        The exchange pair to perform the order for.
    price : float
        The current price.
    down_percent: float
        The percentage the price is allowed to drop before exiting.
    up_percent : float
        The percentage the price is allowed to rise before exiting.
    quantity : `Quantity`
        The quantity of the order.
    portfolio : `Portfolio`
        The portfolio being used in the order.
    start : int, optional
        The start time of the order.
    end : int, optional
        The end time of the order.

    Returns
    -------
    `Order`
        A stop order controlling for the percentages above and below the entry
        price.
    """

    side = TradeSide(side)

    order = Order(
        step=portfolio.clock.step,
        side=side,
        trade_type=TradeType(trade_type),
        exchange_pair=exchange_pair,
        price=price,
        start=start,
        end=end,
        quantity=quantity,
        portfolio=portfolio,
        criteria=DailyTL()
    )

    criteria = (Stop("down", down_percent) ^ Stop("up", up_percent)) & DailyTL()
    risk_management = OrderSpec(
        side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
        trade_type=TradeType.MARKET,
        exchange_pair=exchange_pair,
        criteria=criteria
    )

    order.add_order_spec(risk_management)

    return order
