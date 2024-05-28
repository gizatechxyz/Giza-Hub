from flask import Flask, request, render_template, jsonify
import asyncio
from geiger import motema

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        address = request.form.get('address')
        if address:
            try:
                receipt = asyncio.run(motema(address))
                if receipt is None:
                    return jsonify({'error': 'No receipt received'}), 500
                transaction_data = {
                    "address": str(address),
                    "transaction_hash": receipt.transactionHash.hex(),
                    "status": receipt.status
                }
                return jsonify(transaction_data), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid request'}), 400
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)