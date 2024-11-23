from flask import Flask, request
from BattlesnakePipeline.battlesnake_pipeline import BattlesnakePipeline 

app = Flask(__name__)

HOST = "0.0.0.0"
PORT = 8000

pipeline = BattlesnakePipeline.load('./model_pipeline')

@app.route("/")
def snake_metadata():
    return {
      "apiversion": "1",
      "author": "MyUsername",
      "color": "#888888",
      "head": "default",
      "tail": "default",
      "version": "0.0.1-beta"
    }

@app.route("/start", methods=['POST'])
def start_game():
    return ''
  
@app.route("/move", methods=['POST'])
def perform_move():
  content = request.json
  return {'move':pipeline.predict(content)}
  
if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)