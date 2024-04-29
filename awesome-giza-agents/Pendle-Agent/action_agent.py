from ape import Contract
from ape import accounts
router = Contract("0x00000000005BBB0EF59571E58418F9a4357b68A0")

market = Contract("0xD0354D4e7bCf345fB117cabe41aCaDb724eccCa2")

stETH = Contract("0xae7ab96520DE3A18E5e111B5EaAb095312D7fE84")

uniswap_Routerv3 = Contract("0xE592427A0AEce92De3Edee1F18E0157C05861564")
wETH = Contract("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
wstETH = Contract("0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0")
stETH2 = Contract("0x17144556fd3424EDC8Fc8A4C940B2D04936d17eb")
eETH = Contract("0x35fa164735182de50811e8e2e824cfb9b6118ac2")
eETH_LP = Contract("0x308861A430be4cce5502d0A12724771Fc6DaF216")

weETH = Contract("0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee")

wETH_wstETH_pool = Contract("0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852")



Pt_weETH_Market = Contract("0xF32e58F92e60f4b0A37A69b95d642A471365EAe8")







mint_params = {
        "path": ["0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2","0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0"] ,
        "recipient": dev,
        "deadline": 19710009,
        "amountIn": 5,
        "amountOutMinimum": 3,
    }

ExactInputSingleParams = {
        'tokenIn': "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        'tokenOut':  "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
        'fee': 100,
        'recipient' : acc,
        'deadline' :19810040 ,
        'amountIn': 6*10**18,
        'amountOutMinimum': 0 ,
        'sqrtPriceLimitX96': 0,
    }
 
ExactInputSingleParams = {
        'tokenIn': "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        'tokenOut':  "0x7f39C581F595B53c5cb19bD0b3f8dA6c935E2Ca0",
        'fee': 100,
        'recipient' : acc,
        'deadline' :19810040 ,
        'amountIn': 6*10**18,
        'amountOutMinimum': 0 ,
        'sqrtPriceLimitX96': 0,
    }

DefaultApprox = {
        'guessMin': 0,
        'guessMax': int(1e30) ,
        'guessOffchain': 0,
        'maxIteration' : 256,
        'eps' : 1e14,
}

InputTuple = {
        "tokenIn": "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee" ,
        "netTokenIn": 964170416687097156,
        "tokenMintSy":"0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee" ,
        "pendleSwap": "0x0000000000000000000000000000000000000000",
        "swapData": None,
}
Limit = None


(dev, "0xF32e58F92e60f4b0A37A69b95d642A471365EAe8",0,{
        'guessMin': 0,
        'guessMax': int(1e30) ,
        'guessOffchain': 0,
        'maxIteration' : 256,
        'eps' : 1e14,
},{
        "tokenIn": "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee" ,
        "netTokenIn": 964170416687097156,
        "tokenMintSy":"0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee" ,
        "pendleSwap": "0x0000000000000000000000000000000000000000",
        "swapData": None,
}, None,msg=dev )