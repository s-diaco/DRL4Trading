from itertools import product

from tensortrade.oms.wallets.portfolio import Portfolio
from oms.orders.create import risk_managed_dtl_order

from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.oms.orders.trade import TradeSide
from gym.spaces import Space, Discrete


class DailyTLOrders(ManagedRiskOrders):
    """
    A discrete action scheme for markets with "Daily Trading Limit" and no
    "Short Selling". This is based on "ManagedRiskOrders" from tensortrade.
    """
    @property
    def action_space(self) -> 'Space':
        if not self._action_space:
            self.actions = product(
                self.stop,
                self.take,
                self.trade_sizes,
                self.durations,
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (stop, take, proportion, duration)) = self.actions[action]

        side = TradeSide.BUY

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        params = {
            'side': side,
            'exchange_pair': ep,
            'price': ep.price,
            'quantity': quantity,
            'down_percent': stop,
            'up_percent': take,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_dtl_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]
