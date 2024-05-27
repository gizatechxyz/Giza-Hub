import argparse
import logging
import os
import pprint
from logging import getLogger

import numpy as np
from dotenv import find_dotenv, load_dotenv
from giza.agents import AgentResult, GizaAgent

from addresses import ADDRESSES
from lp_tools import get_tick_range
from uni_helpers import (approve_token, check_allowance, close_position,
                         get_all_user_positions, get_mint_params)

load_dotenv(find_dotenv())

os.environ["DEV_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")

logging.basicConfig(level=logging.INFO)


def process_data(realized_vol: float, dec_price_change: float):
    pct_change_sq = (100 * dec_price_change) ** 2
    X = np.array([[realized_vol, pct_change_sq]])
    return X


def get_data():
    # TODO: implement fetching onchain or from some other source
    # hardcoding the values for now
    realized_vol = 4.20
    dec_price_change = 0.1
    return realized_vol, dec_price_change


def create_agent(
    model_id: int, version_id: int, chain: str, contracts: dict, account: str
):
    """
    Create a Giza agent for the volatility prediction model
    """
    agent = GizaAgent(
        contracts=contracts,
        id=model_id,
        version_id=version_id,
        chain=chain,
        account=account,
    )
    return agent


def predict(agent: GizaAgent, X: np.ndarray):
    """
    Predict the next day volatility.

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    prediction = agent.predict(input_feed={"val": X}, verifiable=True, job_size="XL")
    return prediction


def get_pred_val(prediction: AgentResult):
    """
    Get the value from the prediction.

    Args:
        prediction (dict): Prediction from the model.

    Returns:
        int: Predicted value.
    """
    # This will block the executon until the prediction has generated the proof
    # and the proof has been verified
    return prediction.value[0][0]


def rebalance_lp(
    tokenA_amount: int,
    tokenB_amount: int,
    pred_model_id: int,
    pred_version_id: int,
    account="dev",
    chain=f"ethereum:sepolia:{sepolia_rpc_url}",
    nft_id=None,
):
    logger = getLogger("agent_logger")
    nft_manager_address = ADDRESSES["NonfungiblePositionManager"][11155111]
    tokenA_address = ADDRESSES["UNI"][11155111]
    tokenB_address = ADDRESSES["WETH"][11155111]
    pool_address = "0x287B0e934ed0439E2a7b1d5F0FC25eA2c24b64f7"
    user_address = "0xCBB090699E0664f0F6A4EFbC616f402233718152"
    pool_fee = 3000
    logger.info("Fetching input data")
    realized_vol, dec_price_change = get_data()
    logger.info(f"Input data: {realized_vol}, {dec_price_change}")
    X = process_data(realized_vol, dec_price_change)
    contracts = {
        "nft_manager": nft_manager_address,
        "tokenA": tokenA_address,
        "tokenB": tokenB_address,
        "pool": pool_address,
    }
    agent = create_agent(
        model_id=pred_model_id,
        version_id=pred_version_id,
        chain=chain,
        contracts=contracts,
        account=account,
    )
    result = predict(agent, X)
    predicted_value = get_pred_val(result)
    logger.info(f"Result: {result}")
    with agent.execute() as contracts:
        logger.info("Executing contract")
        if nft_id is None:
            positions = [
                max(get_all_user_positions(contracts.nft_manager, user_address))
            ]
        else:
            positions = [nft_id]
        logger.info(f"Closing the following positions {positions}")
        for nft_id in positions:
            close_position(user_address, contracts.nft_manager, nft_id)
        logger.info("Calculating mint params...")
        _, curr_tick, _, _, _, _, _ = contracts.pool.slot0()
        if not check_allowance(
            contracts.tokenA, nft_manager_address, account, tokenA_amount
        ):
            approve_token(contracts.tokenA, nft_manager_address, tokenA_amount)
        if not check_allowance(
            contracts.tokenB, nft_manager_address, account, tokenB_amount
        ):
            approve_token(contracts.tokenB, nft_manager_address, tokenB_amount)
        tokenA_decimals = contracts.tokenA.decimals()
        tokenB_decimals = contracts.tokenB.decimals()
        predicted_value = predicted_value / 100 * 1.96  # convert to decimal %
        lower_tick, upper_tick = get_tick_range(
            curr_tick, predicted_value, tokenA_decimals, tokenB_decimals, pool_fee
        )
        mint_params = get_mint_params(
            user_address,
            contracts.tokenA.address,
            contracts.tokenB.address,
            tokenA_amount,
            tokenB_amount,
            pool_fee,
            lower_tick,
            upper_tick,
        )
        # step 5: mint new position
        logger.info("Minting new position...")
        contract_result = contracts.nft_manager.mint(mint_params)
        logger.info("SUCCESSFULLY MINTED A POSITION")
        logger.info("Contract executed")

    logger.info(f"Contract result: {contract_result}")
    pprint.pprint(contract_result.__dict__)
    logger.info("Finished")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--model-id", metavar="M", type=int, help="model-id")
    parser.add_argument("--version-id", metavar="V", type=int, help="version-id")
    parser.add_argument("--tokenA-amount", metavar="A", type=int, help="tokenA-amount")
    parser.add_argument("--tokenB-amount", metavar="B", type=int, help="tokenB-amount")

    # Parse arguments
    args = parser.parse_args()

    MODEL_ID = args.model_id
    VERSION_ID = args.version_id
    tokenA_amount = args.tokenA_amount
    tokenB_amount = args.tokenB_amount

    rebalance_lp(tokenA_amount, tokenB_amount, MODEL_ID, VERSION_ID)
