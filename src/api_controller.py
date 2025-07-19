from flask import Flask, jsonify, request
import subprocess

app = Flask(__name__)

@app.route('/run-tremor', methods=['POST'])
def run_tremor_analysis():
    try:
        # Leggi l'header 'patientid'
        patient_id = request.headers.get('patientid')
        if not patient_id:
            return jsonify({'status': 'error', 'message': 'Missing patientid header'}), 400

        # Avvia lo script con patientid come argomento
        result = subprocess.run(
            ['python', 'src/tremor_analisys.py', patient_id],
            capture_output=True,
            text=True
        )
        return jsonify({
            'status': 'success',
            'stdout': 'stdout',
            'stderr': result.stderr,
            'returncode': result.returncode
        }), 200 if result.returncode == 0 else 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)