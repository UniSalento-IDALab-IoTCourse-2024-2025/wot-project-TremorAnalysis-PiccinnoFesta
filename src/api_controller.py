from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run-tremor', methods=['GET'])
def run_tremor_analysis():
    try:
        # Avvia lo script tremor_analisys.py
        result = subprocess.run(
            ['python', 'tremor_analisys.py'],
            capture_output=True,
            text=True
        )
        return jsonify({
            'status': 'success',
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }), 200 if result.returncode == 0 else 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)