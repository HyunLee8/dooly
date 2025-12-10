from flask import Flask, request, jsonify
from flask_cors import CORS
from src.llm_reasoning.mision.logic import get_agent_response
from src.audio_feedback.stt import transcribe_audio

app = Flask(__name__)
CORS(app)

@app.route('/api/agent', methods=['POST'])
def agent():
    data = request.get_json()
    user_req = data.get('user_req')
    
    if not user_req:
        return jsonify({"error": "user_req is required"}), 400
    
    try:
        class_ids = get_agent_response(user_req)
        return jsonify({'class_ids': class_ids})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stt', methods=['POST'])
def speach_to_text():
    if 'audio_file' not in request.files:
        return jsonify({"error": "audio_file is required"}), 400

    audio_file = request.files['audio_file']

    try:
        text = transcribe_audio(audio_file)
        if text is None:
            return jsonify({'error': 'Transcription failed'}), 500
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
