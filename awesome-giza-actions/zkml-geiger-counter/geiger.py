import os
from datetime import datetime
import sys
import time
import pandas as pd
import numpy as np
import asyncio
import torch
import logging
import requests
from web3 import Web3
from pyflipper.pyflipper import PyFlipper
from giza_actions.action import Action, action
from giza_actions.agent import GizaAgent
from giza_actions.task import task
from dotenv import load_dotenv
from eth_account import Account
from eth_typing import Address

load_dotenv()

def import_account(mnemonic):
    account = Account.from_mnemonic(mnemonic)
    return account

# Read CSV files in `data` folder and process the most recent file reads into a numpy tensor
@task
def filter_and_process_data_to_numpy():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the 'data' directory.")

    latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    csv_path = os.path.join(data_dir, latest_csv)
    df = pd.read_csv(csv_path)
    df['cps'] = df['cps'].astype(float)
    cps_values = df['cps'].values

    # Find the top 3 values from the cps_values
    top_3_values = np.sort(cps_values)[-3:][::-1]

    # Create a NumPy tensor with the top 3 values
    return_tensor = np.array(top_3_values, dtype=np.float64)

    return return_tensor

@task
def read_csv_from_flipper():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Reading files from Flipper... 📖")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    try:
        flipper = PyFlipper(com="/dev/cu.usbmodemflip_Anen1x1")
    except Exception as e:
        print(f"Error connecting to Flipper, continuing...: {e}")
        pass

    files_and_dirs = flipper.storage.list(path="/ext")
    logger.info(f"Files and directories found on Flipper: {files_and_dirs}")

    for file_dict in files_and_dirs.get('files', []):
        file_name = file_dict['name']
        file_path = f"/ext/{file_name}"
        logger.info(f"Reading {file_path} from Flipper...")
        try:
            file_data = flipper.storage.read(file=file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

        logger.info(f"Read {file_name} from Flipper. Saving to {data_dir}...")
        filename = os.path.join(data_dir, file_name)
        try:
            with open(filename, 'w') as f:
                f.write(file_data)
                logger.info(f"Saved {file_name} to {filename}")
        except Exception as e:
            logger.error(f"Error saving {file_name} to {filename}: {e}")

    # Find the most recent CSV file
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        most_recent_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
        logger.info(f"Most recent CSV file: {most_recent_csv}")
        csv_path = os.path.join(data_dir, most_recent_csv)
        try:
            with open(csv_path, 'r') as f:
                csv_data = f.read()
                logger.info(f"Contents of {most_recent_csv}:\n{csv_data}")
        except Exception as e:
            logger.error(f"Error reading {most_recent_csv}: {e}")
    else:
        logger.info("No CSV files found in the data directory.")

@action
async def motema(address):
    try:
        address = Web3.to_checksum_address(address)
    except ValueError as e:
        raise ValueError(f"Invalid address format: {e}")

    print("Address properly parsed. Starting Motema flow... 🩵")
    print("Address: ", address)
    time.sleep(20)

    read_csv_from_flipper()
    tensor = filter_and_process_data_to_numpy()
    print("Tensor: ", tensor)

    Account.enable_unaudited_hdwallet_features()
    mnemonic = os.getenv("MNEMONIC")
    account = import_account(mnemonic)
    print("Account address: ", account.address)

    # Create GizaAgent instance
    model_id = 430
    version_id = 1
    agent = GizaAgent(id=model_id, version=version_id)

    # Run and saveinf erence
    agent.infer(input_feed={"tensor_input": tensor}, job_size="S")
    
    # Get proof
    proof, proof_path = agent.get_model_data()

    # Verify proof
    verified = await agent.verify(proof_path)
    
    # verified = True
    mark = False

    if verified:
        print("Proof verified. 🚀")
        # The threshold relu function will set all values less than the threshold to 0
        print("Inference: ", agent.inference)
        if any(x >= 0 for x in agent.inference):
            mark = True
        else:
            pass
        if mark is True:
            print("This person has been exposed to radiation. Let's get them a payment.")
            signed_proof, is_none, proof_message, signable_proof_message = agent.sign_proof(account, proof, proof_path)
            rpc = os.getenv("ALCHEMY_URL")

            # Get contract address
            contract_address = Web3.to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
            print ("Contract address: ", contract_address)
            print("Transaction being sent from: ", account.address)
            # Transmit transaction
            receipt = await agent.transmit(
                account=account,
                contract_address=contract_address,
                chain_id=11155111,
                abi_path="contracts/abi/MotemaPoolAbi.json",
                function_name="claim",
                params=[address],
                value=None,
                # todo: make this an option
                signed_proof=signed_proof,
                is_none=is_none,
                proofMessage=proof_message,
                signedProofMessage=signable_proof_message,
                rpc_url=rpc,
                unsafe=False
            )
            print("Receipt: ", receipt)
            return receipt
        else:
            raise Exception("It doesn't seem like you've been in the mines.")
    else:
        raise Exception("Proof verification failed.")
    
async def main(address):
    await motema(address)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide an address as a command-line argument.")
        sys.exit(1)

    address = sys.argv[1]
    asyncio.run(main(address))