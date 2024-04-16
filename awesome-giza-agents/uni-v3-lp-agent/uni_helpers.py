from ape import Contract, accounts, networks, chain
import time
from addresses import ADDRESSES
from lp_tools import *
import os
from giza_actions.task import task
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

dev_passphrase = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")


@task(name="Check allowance")
def check_allowance(token, spender, account, amount):
    return token.allowance(account, spender) >= amount


@task(name="Approve token spend")
def approve_token(token, spender, amount):
    return token.approve(spender, amount)


@task(name="Get mint parameters")
def get_mint_params(
    user_address,
    tokenA_address,
    tokenB_address,
    amount0,
    amount1,
    pool_fee,
    lower_tick,
    upper_tick,
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


@task(name="Get all user LP positions")
def get_all_user_positions(nft_manager, user_address):
    n_positions = nft_manager.balanceOf(user_address)
    positions = []
    for n in range(n_positions):
        position = nft_manager.tokenOfOwnerByIndex(user_address, n)
        positions.append(position)
    return positions


# @task(name="Get the position liquidity")
def get_pos_liquidity(nft_manager, nft_id):
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


@task(name=f"Close the position")
def close_position(user_address, nft_manager, nft_id):
    liq = get_pos_liquidity(nft_manager, nft_id)
    if liq > 0:
        nft_manager.decreaseLiquidity((nft_id, liq, 0, 0, int(time.time() + 60)))
        nft_manager.collect((nft_id, user_address, MAX_UINT_128, MAX_UINT_128))


if __name__ == "__main__":
    networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()
    chain_id = chain.chain_id

    # step 1: set params
    tokenA_amount = 1000
    tokenB_amount = 1000
    pct_dev = 0.1
    pool_fee = 3000
    # step 2: load contracts
    tokenA = Contract(ADDRESSES["UNI"][chain_id])
    tokenB = Contract(ADDRESSES["WETH"][chain_id])
    nft_manager = Contract(ADDRESSES["NonfungiblePositionManager"][chain_id])
    pool_factory = Contract(ADDRESSES["PoolFactory"][chain_id])
    pool_address = pool_factory.getPool(tokenA.address, tokenB.address, pool_fee)
    pool = Contract(pool_address)
    dev = accounts.load("dev")
    dev.set_autosign(True, passphrase=dev_passphrase)
    user_address = dev.address
    with accounts.use_sender("dev"):
        # step 3: fetch open positions
        positions = get_all_user_positions(nft_manager, user_address)
        print(f"Found the following open positions: {positions}")
        # step 4: close all positions
        print("Closing all open positions...")
        for nft_id in positions:
            close_position(user_address, nft_manager, nft_id)
        # step 4: calculate mint params
        print("Calculating mint params...")
        _, curr_tick, _, _, _, _, _ = pool.slot0()
        tokenA_decimals = tokenA.decimals()
        tokenB_decimals = tokenB.decimals()
        curr_price = tick_to_price(curr_tick, tokenA_decimals, tokenB_decimals)
        lower_tick, upper_tick = get_tick_range(
            curr_tick, pct_dev, tokenA_decimals, tokenB_decimals, pool_fee
        )
        mint_params = get_mint_params(
            user_address, tokenA_amount, tokenB_amount, pool_fee, lower_tick, upper_tick
        )
        # step 5: mint new position
        print("Minting new position...")
        nft_manager.mint(mint_params)
