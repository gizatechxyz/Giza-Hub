import math

MIN_TICK = -887272
MAX_TICK = -MIN_TICK
TICKS_Q = 1.0001
Q96 = 2**96
MAX_UINT_128 = 2 ** (128) - 1
_tick_spacing = {100: 1, 500: 10, 3_000: 60, 10_000: 200}

# https://ethereum.stackexchange.com/questions/150280/calculate-amount-of-eth-and-usdc-after-minting-a-position-in-uniswap-v3


def price_to_tick(price):
    sqrtPriceX96 = int(price * 2**96)
    tick = math.floor(math.log((sqrtPriceX96 / Q96) ** 2) / math.log(TICKS_Q))
    return tick


def tick_to_price(tick, decimals0, decimals1, invert=False):
    if invert:
        return 1 / (TICKS_Q**tick / 10 ** (decimals1 - decimals0))
    else:
        return TICKS_Q**tick / 10 ** (decimals1 - decimals0)


def get_min_tick(fee: int):
    min_tick_spacing: int = _tick_spacing[fee]
    return -(MIN_TICK // -min_tick_spacing) * min_tick_spacing


def get_max_tick(fee: int):
    max_tick_spacing: int = _tick_spacing[fee]
    return (MAX_TICK // max_tick_spacing) * max_tick_spacing


def default_tick_range(fee: int):
    min_tick = get_min_tick(fee)
    max_tick = get_max_tick(fee)
    return min_tick, max_tick


def nearest_tick(tick: int, fee: int):
    min_tick, max_tick = default_tick_range(fee)
    assert (
        min_tick <= tick <= max_tick
    ), f"Provided tick is out of bounds: {(min_tick, max_tick)}"
    tick_spacing = _tick_spacing[fee]
    rounded_tick_spacing = round(tick / tick_spacing) * tick_spacing
    if rounded_tick_spacing < min_tick:
        return rounded_tick_spacing + tick_spacing
    elif rounded_tick_spacing > max_tick:
        return rounded_tick_spacing - tick_spacing
    else:
        return rounded_tick_spacing


def get_tick_range(curr_tick, pct_dev, tokenA_decimals, tokenB_decimals, fee):
    curr_price = tick_to_price(curr_tick, tokenA_decimals, tokenB_decimals)
    upper_price = curr_price * (1 + pct_dev)
    lower_price = curr_price * (1 - pct_dev)
    lower_tick = price_to_tick(lower_price)
    upper_tick = price_to_tick(upper_price)
    lower_tick = nearest_tick(lower_tick, fee)
    upper_tick = nearest_tick(upper_tick, fee)
    return lower_tick, upper_tick
