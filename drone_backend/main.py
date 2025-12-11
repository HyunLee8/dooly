from flask import Flask, request, jsonify
from flask_cors import CORS
from src.llm_reasoning.mision.logic import get_agent_response
from src.audio_feedback.stt import transcribe_audio

app = Flask(__name__)
CORS(app)

#PROCESS ROUTE HANDLES BOTH IN ONE GO - MAYBE USE THIS LATER FOR SMT DIFF

"""
@app.route('/api/agent', methods=['POST'])
def agent():
    data = request.get_json()
    user_req = data.get('user_req')
    try:
        class_ids, message = get_agent_response(user_req) 
        return jsonify({'class_ids': class_ids,
                        'message': message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stt', methods=['POST'])
def speach_to_text():
    audio_file = request.files['audio_file']

    try:
        text = transcribe_audio(audio_file)
        if text is None:
            return jsonify({'error': 'Transcription failed'}), 500
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

@app.route('/api/process', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    transcription = transcribe_audio(audio_file)
    if not transcription:
        return jsonify({'message': 'I didn\'t hear anything. Please try again.'})
    class_ids, message = get_agent_response(transcription)
    return jsonify({'transcription': transcription, 'class_ids': class_ids, 'message': message})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
