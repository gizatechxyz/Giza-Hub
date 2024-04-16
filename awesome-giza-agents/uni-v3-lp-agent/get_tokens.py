import os

from addresses import ADDRESSES
from ape import Contract, accounts, chain, networks
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

dev_passphrase = os.environ.get("DEV_PASSPHRASE")
sepolia_rpc_url = os.environ.get("SEPOLIA_RPC_URL")

if __name__ == "__main__":
    networks.parse_network_choice(f"ethereum:sepolia:{sepolia_rpc_url}").__enter__()
    chain_id = chain.chain_id
    weth_mint_amount = 0.0001
    pool_fee = 3000
    uni = Contract(ADDRESSES["UNI"][chain_id])
    weth = Contract(ADDRESSES["WETH"][chain_id])
    weth_decimals = weth.decimals()
    uni_decimals = uni.decimals()
    weth_mint_amount = int(weth_mint_amount * 10**weth_decimals)
    uni_mint_amount = int(0.5 * weth_mint_amount)

    pool_factory = Contract(ADDRESSES["PoolFactory"][chain_id])
    pool_address = "0x287B0e934ed0439E2a7b1d5F0FC25eA2c24b64f7"
    pool = Contract(pool_address)
    swap_router = Contract(ADDRESSES["Router"][chain_id])
    wallet = accounts.load("dev")
    wallet.set_autosign(True, passphrase=dev_passphrase)
    with accounts.use_sender("dev"):
        print(f"Minting {weth_mint_amount/10**weth_decimals} WETH")
        weth.deposit(value=weth_mint_amount)
        print("Approving WETH for swap")
        weth.approve(swap_router.address, weth_mint_amount)
        swap_params = {
            "tokenIn": weth.address,
            "tokenOut": uni.address,
            "fee": pool_fee,
            "recipient": wallet.address,
            "amountIn": weth_mint_amount,
            "amountOutMinimum": 0,
            "sqrtPriceLimitX96": 0,
        }
        swap_params = tuple(swap_params.values())
        print("Swapping WETH for UNI")
        amountOut = swap_router.exactInputSingle(swap_params)
        print(f"Successfully minted {uni_mint_amount/10**uni_decimals} UNI")

    print(f"Your WETH balance: {weth.balanceOf(wallet.address)/10**weth_decimals}")
    print(f"Your UNI balance: {uni.balanceOf(wallet.address)/10**uni_decimals}")
