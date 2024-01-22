import os
from werkzeug.utils import secure_filename
from src.logger import logging
from src.exception import CustomException

from src.components.pipeline.predict_pipeline import PredictPipeline
from flask import render_template, request, url_for, Flask, redirect

app = Flask(__name__, static_url_path="/static")
uploads_dir = os.path.join(os.getcwd(), "static", "uploads")
predict_pipeline_obj = PredictPipeline()


def get_prediction(predict_pipeline_obj, image_name):
    return predict_pipeline_obj.predict(image_name)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image_file"]
        image_file_name = secure_filename(image_file.filename)
        image_path = os.path.join(uploads_dir, image_file_name)
        image_file.save(image_path)
        image_path = os.path.join("static", os.path.join("uploads", image_file_name))
        caption = get_prediction(predict_pipeline_obj, image_file_name)
        return render_template(
            "predict.html",
            caption=caption,
            image_path=image_path,
            file_name=image_file_name,
        )

    return render_template(
        "predict.html",
        caption="Upload Image. :)",
        image_path="image_placeholder.png",
        file_name="image_placeholder.png",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
