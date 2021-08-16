from tensortrade.feed.core.base import Stream
from tensortrade.oms.exchanges.exchange import Exchange
from tensortrade.oms.orders.criteria import Criteria
from tensortrade.oms.orders.order import Order
from tensortrade.oms.orders.trade import TradeSide


class DailyTL(Criteria):
    """An order criteria that allows execution when daily trading limit allows.
    """

    def check(self, order: 'Order', exchange: 'Exchange') -> bool:
        b_queue = Stream.select(
            exchange.streams(),
            lambda s: s.name == f'{exchange.name}:/bqueue-{order.pair}').value
        s_queue = Stream.select(
            exchange.streams(),
            lambda s: s.name == f'{exchange.name}:/squeue-{order.pair}').value
        stopped = Stream.select(
            exchange.streams(),
            lambda s: s.name == f'{exchange.name}:/stopped-{order.pair}').value
        buy_satisfied = (order.side == TradeSide.BUY and not b_queue)
        sell_satisfied = (order.side == TradeSide.SELL and not s_queue)
        dtl_satisfied = (buy_satisfied or sell_satisfied) and not stopped
        return dtl_satisfied

    def __str__(self) -> str:
        return "<Daily Trading Limit>"
