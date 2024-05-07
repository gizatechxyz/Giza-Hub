# Introduction

In this tutorial, we will go over the basics of Giza agents and implement a Pendle Trading bot using the Giza Agents Framework. Altough we will explain the basic flow to create an agent, we strongly recommend you to first read the Giza Agents[] and Giza Actions SDK[] documentation to understand the fundementals of Giza Stack.

This tutorial is meant to exemplify various ways the ML models can be used within the Giza Agents framework to create on-chain actions in an automated way, while still adhering to the trust-minimizing approach that underly blockchains and DeFi. Pendle Protocol is a great protocol to build on since it is relatively popular and is a great enviroment to build a yield trading bot given its native yield abstraction tokens, however similar agents can be built on any DeFi protocol.

In addition, unlike Uniswap V3 and MNIST agents, the agent is built in a local, forked Ethereum Network using Foundry. In the following sections, we will illustrate how to use a local forked network to develop an agent in a safe testing enviroment.

*It is important to underscore that this agent is a proof of work that is built in a forked, local Ethereum network. This agent is not audited and should not be deployed onchain without heavy caution.*


# 1. Pendle Protocol Primer

[Pendle Documentation](https://docs.pendle.finance/Introduction)

Pendle Protocol is a permissionless yield trading protocol that use tokenization to seperate yield from the underlying token. To briefly summarize, any yield bearing token (in our project, we use weETH) can be wrapped into its standardized SY Pendle variant (SY-weETH). This SY token, which represents both the underlying non-yield token (ETH) as well as its yield bearing portion, can then be seperated and traded with two tokenized components, which are called Principal Token (PT-weETH) and Yield Token (YT-weETH). Through its PT-SY AMM's, Pendle Protocol allows users to trade between the potential yield and the underlying value of the token and consequently create yield markets for many popular yield bearing tokens. 

This agent focuses on the weETH, the wrapped version of the ether.fi token, which is traded between its SY and PT variants. To execute informed and profitable trades, agent leverages a yield prediction ZKML model for eETH, verifies its output using the its ZKML proof and ultimately compares it with the PT-SY price as well as the fixed PT-yield to automatically trade between PT-weETH and SY-weETH tokens.

# 2. Installing Dependencies and Project Setup

- Python 3.11 or later must be installed, we recommend using a virtual enviroment.

This project uses poetry as the dependency manager, to install the required dependencies simply execute:

```bash 
$ poetry install
```
- An active Giza account is required to deploy the model and use agents. If you don't have one, you can create one [here](https://cli.gizatech.xyz/examples/basic).

- You also need an ape account to use Giza Agents, you can read how to create an Ape account, as well as the basics of the Ape Framework [here](https://agents.gizatech.xyz/how-to-guides/create-an-account-wallet)

- To run a forked Ethereum network locally, you need to install [Foundry](https://book.getfoundry.sh/getting-started/installation).

- Finally, we will need some environment variables. Create a .env file in the directory of this project and populate it one variable:

```
DEV_PASSPHRASE="<YOUR-APE-ACCOUNT-PASSWORD>"
```

# 3. Yield Prediction Model

This project uses a relatively simple, 3 layered neural network model to predict the yield (APR) of the eETH token after 7 days. You can use the [yield_prediction_model.ipynb notebook](model/yield_prediction_model.ipynb) to follow the implementation of the model step-by-step. In the end of the notebook, the developed model is exported in .onnx format, which will then be used within the Giza CLI to create the ZKML endpoint.

# 4. Model transpilation and Agent creation

Take a look at the [yield_prediction_actions_notebook.ipynb](model/yield_prediction_actions_notebook.ipynb) and follow the steps to learn how to create an agent (as well as learn the fundementals of Giza CLI step-by-step)

Checklist: 

- Create a Giza User
- Transpile the model using Orion
- Create a Giza Workspace
- Deploy your ZKML model into your workspace
- Create an Endpoint using the ZKML model
- Create a Giza Agent that listens to that Endpoint

# 5. Creating a local fork of Ethereum Mainnet, Agent setup

*We are assumign here that you have already installed Foundry*

Open up a new terminal, and type in the following command

```bash 
$ anvil --fork-url <RPC_URL> --fork-block-number 19754466 --fork-chain-id 1 --chain-id 1
```

This creates a local Ethereum mainnet network (chain-id 1) from the 19754466'th block, which we will use to run our agent on. Working on a local fork provides various advantages, such as being able to experiment with any smart contract and protocol that is on the target network.

To be able to use the agent, we require some tokens to begin with.

*Before running the setup.py, make sure to edit the marked lines to include your Ape password and your Ape username*

```bash 
$ python agent/setup.py
```

This command will login to your Ape wallet, and mint you the required amount of weETH to be able to run the agent.


# 6. Pendle Agent

Before running the agent, lets look at some code snippets to understand what the code is actually doing.

```python
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--agent-id", metavar="A", type=int, help="model-id")
    parser.add_argument("--weETH-amount", metavar="W", type=float, help="weETH-amount")
    parser.add_argument("--fixed-yield", metavar="Y", type=float, help="fixed-yield")
    parser.add_argument("--expiration-days", metavar="E", type=int, help="days-untill-expiration")


    # Parse arguments
    args = parser.parse_args()


    agent_id = args.agent_id
    weETH_amount = args.weETH_amount
    fixed_yield = args.fixed_yield
    expiration_days = args.expiration_days


    SY_PY_swap(weETH_amount,agent_id,fixed_yield, expiration_days)
```

To run the agent, we need to type as arguments the id of the agent (in our case : 5), the weETH-amount that we locate to the wallet (lets give 5 weETH), the fixed yield and the expiration date of the pool (these two can be parsed from the pools in a later iteration). The main function simply parses the arguments, and runs the main function of the agent, SY_PY_swap().


```python
    contracts = {
        "router": router,
        "routerProxy": routerProxy,
        "SY_weETH_Market": SY_weETH_Market,
        "weETH": weETH,
        "PT_weETH": PT_weETH,
    }
    agent = create_agent(
        agent_id=agent_id,
        chain=chain,
        contracts=contracts,
        account_alias=account,
    )
```

We are putting all the contracts that our agents will interact in a dictionary, and using that in agent creation. Giza Agents automatically creates the Contracts() objects you might be familiar from the Ape Framework. 

```python
    with agent.execute() as contracts:
        logger.info("Verification complete, executing contract")

        decimals = contracts.weETH.decimals()
        weETH_amount = weETH_amount * 10**decimals

        state = contracts.SY_weETH_Market.readState(contracts.router.address)

        PT_price = calculate_price(state.lastLnImpliedRate, decimals)
        logger.info(f"Calculated Price: {PT_price}")

        traded_SY_amount, PT_weight = swap_logic(weETH_amount, PT_price, fixed_yield, result.value[0][0], expiration_days)

        logger.info(f"The amount of SY to be traded: {traded_SY_amount}, PT_weight: {PT_weight}")


        contracts.weETH.approve(contracts.routerProxy.address, traded_SY_amount , max_fee = 10**10)

        contracts.routerProxy.swapExactTokenForPt(wallet_address, contracts.SY_weETH_Market.address,0,guess_out_tuple()
                                   ,input_tuple(contracts.weETH, traded_SY_amount),no_limit_order_params())
        
        PT_balance = contracts.PT_weETH.balanceOf(wallet_address)
        weETH_balance = contracts.weETH.balanceOf(wallet_address)


        logger.info(f"Swap succesfull! Currently, you own: {PT_balance} PT-weETH and {weETH_balance} weETH")
```

The lines starting from agent.execute() represents the onchain interactions that take place. As you can see, we easily access the functions of the contracts we have given as input to the agent by using their dictionary keys. We calculate the price of the PT_SY swap, and then calculate how much we want to sell SY to buy PT with the swap_logic() function. We approve the tokens exchange, and then swap the calculated amounts of SY tokens for the PT tokens.


# 7. Running the Pendle Agent

To execute the agent, run the following command

```bash
python agent/agent.py --agent-id 4 --weETH-amount 5 --fixed-yield 1.2 --expiration-days 30
```
 *You can change the variables to see how it affects the trade at the end.*


```
15:53:46.145 | INFO    | Created flow run 'imperial-sloth' for flow 'SY-PY-swap'
15:53:46.145 | INFO    | Action run 'imperial-sloth' - View at https://actions-server-ege-dblzzhtf5q-ew.a.run.app/flow-runs/flow-run/6769434d-7234-4540-8980-b39fb40c477e
15:53:46.765 | INFO    | Action run 'imperial-sloth' - Created task run 'Create a Giza agent using Agent_ID-0' for task 'Create a Giza agent using Agent_ID'
15:53:46.765 | INFO    | Action run 'imperial-sloth' - Executing 'Create a Giza agent using Agent_ID-0' immediately...
15:53:48.583 | INFO    | Task run 'Create a Giza agent using Agent_ID-0' - Finished in state Completed()
15:53:48.673 | INFO    | Action run 'imperial-sloth' - Created task run 'Run the yield prediction model-0' for task 'Run the yield prediction model'
15:53:48.673 | INFO    | Action run 'imperial-sloth' - Executing 'Run the yield prediction model-0' immediately...
🚀 Starting deserialization process...
✅ Deserialization completed! 🎉
15:53:53.102 | INFO    | Task run 'Run the yield prediction model-0' - Finished in state Completed()
15:53:53.104 | INFO    | Action run 'imperial-sloth' - Result: AgentResult(input={'input': array([[0.9040785 , 0.93346555, 0.16330279, 0.00332454, 0.31772107,
        0.02796276, 1.01916988]])}, request_id=4e8cc39af7a443a0af66b0426cb6b62b, value=[[0.05644226]])
INFO: Connecting to existing Erigon node at https://ethereum-rpc.publicnode.com/[hidden].
15:53:53.668 | INFO | Connecting to existing Erigon node at https://ethereum-rpc.publicnode.com/[hidden].
WARNING: Danger! This account will now sign any transaction it's given.
15:53:54.638 | WARNING | Danger! This account will now sign any transaction it's given.
15:53:54.649 | INFO    | Action run 'imperial-sloth' - Verification complete, executing contract
15:53:56.089 | INFO    | Action run 'imperial-sloth' - Calculated Price: 1.3187606552347648
15:56:41.745 | INFO    | Action run 'imperial-sloth' - The amount of SY to be traded: 1554221596960899584, PT_weight: 0.31084431939217994
WARNING: Using cached key for pendle-agent
15:56:41.765 | WARNING | Using cached key for pendle-agent
INFO: Confirmed 0xfaa27a6370eacd3123baa95fb5f9597a8385897be935e818d719dbc8080c270d (total fees paid = 340844006459177)
15:56:42.123 | INFO | Confirmed 0xfaa27a6370eacd3123baa95fb5f9597a8385897be935e818d719dbc8080c270d (total fees paid = 340844006459177)
INFO: Confirmed 0xfaa27a6370eacd3123baa95fb5f9597a8385897be935e818d719dbc8080c270d (total fees paid = 340844006459177)
15:56:42.130 | INFO | Confirmed 0xfaa27a6370eacd3123baa95fb5f9597a8385897be935e818d719dbc8080c270d (total fees paid = 340844006459177)
WARNING: Using cached key for pendle-agent
15:56:42.457 | WARNING | Using cached key for pendle-agent
INFO: Confirmed 0xb37d84466093365255b4dd3e6a90bbb4b0fd5f38a5b8ee19449f385f9d3531fa (total fees paid = 2419237719668910)
15:56:46.165 | INFO | Confirmed 0xb37d84466093365255b4dd3e6a90bbb4b0fd5f38a5b8ee19449f385f9d3531fa (total fees paid = 2419237719668910)
INFO: Confirmed 0xb37d84466093365255b4dd3e6a90bbb4b0fd5f38a5b8ee19449f385f9d3531fa (total fees paid = 2419237719668910)
15:56:46.174 | INFO | Confirmed 0xb37d84466093365255b4dd3e6a90bbb4b0fd5f38a5b8ee19449f385f9d3531fa (total fees paid = 2419237719668910)
15:56:46.186 | INFO    | Action run 'imperial-sloth' - Swap succesfull! Currently, you own: 1685519953369509120 PT-weETH and 6159141736535877677 weETH
15:56:46.398 | INFO    | Action run 'imperial-sloth' - Finished in state Completed('All states completed.')
```

Congrats, you have just used an Giza Agent to provide trade on Pendle Protocol using a ZKML yield prediction model.


