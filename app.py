from flask import  Flask,request,render_template,jsonify
from SRC.pipeline.prediction_pipeline import CustomData,PredictPipeline


app=Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")







if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)