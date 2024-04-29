import os

from addresses import ADDRESSES
from ape import Contract, accounts, chain, networks
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

dev_passphrase = os.environ.get("DEV_PASSPHRASE")
network = "ethereum:mainnet_fork:foundry"

if __name__ == "__main__":
    networks.parse_network_choice(network).__enter__()
    
    eETH = Contract(ADDRESSES["eETH"])
    weETH = Contract(ADDRESSES["weETH"])
    eETH_decimals = eETH.decimals()
    weETH_decimals = weETH.decimals()
    eETH_LP = Contract(ADDRESSES["eETH_LP"])


    dev = accounts.load("pendle-agent")
    dev.set_autosign(True, passphrase=dev_passphrase)
    dev.balance += 10 * int(1e18)

    eETH_mint_amount = 5 * (10**eETH_decimals)

    with accounts.use_sender("pendle-agent"):
        print(f"Staking Ether to get  {eETH_mint_amount/10**eETH_decimals} eETH")
        eETH_LP.deposit(value= eETH_mint_amount)
        print(f"Approving eETH to wrap to weETH")
        eETH.approve(weETH.address,eETH_mint_amount)
        weETH.wrap(eETH_mint_amount)
        weETH_balance = weETH.balanceOf(dev)

    print(f"Dev Wallet has a balance of {weETH_balance/10**weETH_decimals} weETH")



