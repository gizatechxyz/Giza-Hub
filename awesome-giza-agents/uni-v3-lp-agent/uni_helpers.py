import os
import time

from ape.contracts.base import ContractInstance
from dotenv import find_dotenv, load_dotenv
from lp_tools import MAX_UINT_128

load_dotenv(find_dotenv())

dev_passphrase = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")


def check_allowance(token: ContractInstance, spender: str, account: str, amount: int):
    return token.allowance(account, spender) >= amount


def approve_token(token: ContractInstance, spender: str, amount: int):
    return token.approve(spender, amount)


def get_mint_params(
    user_address: str,
    tokenA_address: str,
    tokenB_address: str,
    amount0: int,
    amount1: int,
    pool_fee: int,
    lower_tick: int,
    upper_tick: int,
    deadline=None,
    slippage_tolerance=0.01,
):
    if deadline is None:
        deadline = int(time.time()) + 60
    mint_params = {
        "token0": tokenA_address,
        "token1": tokenB_address,
        "fee": pool_fee,
        "tickLower": lower_tick,
        "tickUpper": upper_tick,
        "amount0Desired": amount0,
        "amount1Desired": amount1,
        "amount0Min": 0,  # int(amount0 * (1 - slippage_tolerance)),
        "amount1Min": 0,  # int(amount1 * (1 - slippage_tolerance)),
        "recipient": user_address,
        "deadline": deadline,
    }
    return tuple(mint_params.values())


def get_all_user_positions(nft_manager: ContractInstance, user_address: str):
    n_positions = nft_manager.balanceOf(user_address)
    positions = []
    for n in range(n_positions):
        position = nft_manager.tokenOfOwnerByIndex(user_address, n)
        positions.append(position)
    return positions


def get_pos_liquidity(nft_manager: ContractInstance, nft_id: int):
    (
        nonce,
        operator,
        token0,
        token1,
        fee,
        tickLower,
        tickUpper,
        liquidity,
        feeGrowthInside0LastX128,
        feeGrowthInside1LastX128,
        tokensOwed0,
        tokensOwed1,
    ) = nft_manager.positions(nft_id)
    return liquidity


def close_position(user_address: str, nft_manager: ContractInstance, nft_id: int):
    liq = get_pos_liquidity(nft_manager, nft_id)
    if liq > 0:
        nft_manager.decreaseLiquidity((nft_id, liq, 0, 0, int(time.time() + 60)))
        nft_manager.collect((nft_id, user_address, MAX_UINT_128, MAX_UINT_128))
