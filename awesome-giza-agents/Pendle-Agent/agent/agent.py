import argparse
import os

import numpy as np
from addresses import ADDRESSES
from dotenv import find_dotenv, load_dotenv
from giza.agents import AgentResult, GizaAgent
from helpers import (calculate_price, guess_out_tuple, input_tuple,
                     no_limit_order_params, swap_logic)
from logging import getLogger

from ape import accounts


load_dotenv(find_dotenv())

os.environ["PENDLE-AGENT_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")




def create_agent(agent_id: int, chain: str, contracts: dict, account_alias: str):
    """
    Create a Giza agent for the Pendle protocol
    """
    agent = GizaAgent.from_id(
        id=agent_id,
        contracts=contracts,
        chain=chain,
        account=account_alias,
    )
    return agent


def predict(agent: GizaAgent, X: np.ndarray):
    """
    Predict the APR one week later.

    Args:
        X (np.ndarray): Input to the model.

    Returns:
        int: Predicted value.
    """
    X = X.reshape(1, 7)
    prediction = agent.predict(input_feed={"input": X}, verifiable=True, job_size="XL")

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
    return prediction.value


def SY_PY_swap(
    weETH_amount: float,
    agent_id: int,
    fixed_yield: float,
    expiration_days: int,
    account="pendle-agent",
    chain="ethereum:mainnet-fork:foundry",
):
    ## Change the PENDLE-AGENT_PASSPHRASE to be {AGENT-NAME}_PASSPHRASE
    os.environ["PENDLE-AGENT_PASSPHRASE"] = os.environ.get("DEV_PASSPHRASE")

    # Create logger
    logger = getLogger("agent_logger")

    # Load the addresses
    router = ADDRESSES["Pendle_v3_Router"]
    routerProxy = ADDRESSES["Pendle_v3_Router_Proxy"]
    SY_weETH_Market = ADDRESSES["SY_weETH_Market"]
    weETH = ADDRESSES["weETH"]
    PT_weETH = ADDRESSES["PT_weETH"]
    wallet_address = accounts.load(account).address
    
    # Load the data, this can be changed to retrieve live data
    file_path = "data/data_array.npy"
    X = np.load(file_path)

    # Fill this contracts dictionary with the contract addresses that our agent will interact with
    contracts = {
        "router": router,
        "routerProxy": routerProxy,
        "SY_weETH_Market": SY_weETH_Market,
        "weETH": weETH,
        "PT_weETH": PT_weETH,
    }
    # Create the agent
    agent = create_agent(
        agent_id=agent_id,
        chain=chain,
        contracts=contracts,
        account_alias=account,
    )
    result = predict(agent, X)

    # If you want to wait until the verification to continue, uncomment the two following lines
    # predicted_value = get_pred_val(result)
    # logger.info("Verification complete, executing contract")
    logger.warning(f"Result: {result}")



    with agent.execute() as contracts:
        logger.warning("Verification complete, executing contract")

        decimals = contracts.weETH.decimals()
        weETH_amount = weETH_amount * 10**decimals

        state = contracts.SY_weETH_Market.readState(contracts.router.address)

        PT_price = calculate_price(state.lastLnImpliedRate, decimals)
        logger.warning(f"Calculated Price: {PT_price}")

        # If the two lines above are not commented, swap_logic will take a while, since it will wait until the result is verified to access result.value
        traded_SY_amount, PT_weight = swap_logic(
            weETH_amount, PT_price, fixed_yield, result.value[0][0], expiration_days
        )

        logger.warning(
            f"The amount of SY to be traded: {traded_SY_amount}, PT_weight: {PT_weight}"
        )

        contracts.weETH.approve(
            contracts.routerProxy.address, traded_SY_amount, max_fee=10**10
        )

        contracts.routerProxy.swapExactTokenForPt(
            wallet_address,
            contracts.SY_weETH_Market.address,
            0,
            guess_out_tuple(),
            input_tuple(contracts.weETH, traded_SY_amount),
            no_limit_order_params(),
        )

        PT_balance = contracts.PT_weETH.balanceOf(wallet_address)
        weETH_balance = contracts.weETH.balanceOf(wallet_address)

        logger.warning(
            f"Swap succesfull! Currently, you own: {PT_balance} PT-weETH and {weETH_balance} weETH"
        )


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--agent-id", metavar="A", type=int, help="model-id")
    parser.add_argument("--weETH-amount", metavar="W", type=float, help="weETH-amount")
    parser.add_argument("--fixed-yield", metavar="Y", type=float, help="fixed-yield")
    parser.add_argument(
        "--expiration-days", metavar="E", type=int, help="days-untill-expiration"
    )

    # Parse arguments
    args = parser.parse_args()

    agent_id = args.agent_id
    weETH_amount = args.weETH_amount
    fixed_yield = args.fixed_yield
    expiration_days = args.expiration_days

    SY_PY_swap(weETH_amount, agent_id, fixed_yield, expiration_days)
