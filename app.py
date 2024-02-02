import os
from werkzeug.utils import secure_filename
from src.logger import logging
from src.exception import CustomException

# from src.components.pipeline.predict_pipeline import PredictPipeline
from flask import render_template, request, url_for, Flask, redirect, flash

app = Flask(__name__, static_url_path="/static")
uploads_dir = os.path.join(os.getcwd(), "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

image_info = {"is_uploaded": False}
# predict_pipeline_obj = PredictPipeline()


def get_prediction(predict_pipeline_obj, image_name):
    return predict_pipeline_obj.predict(image_name)


def is_file_allowed(filename):
    if "." in filename and filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS:
        return True
    return False


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image_file"]
        if not image_file:
            return redirect(url_for("index"))

        image_file_name = secure_filename(image_file.filename)
        if not image_file_name:
            return redirect(url_for("index"))

        if image_file and is_file_allowed(image_file_name):
            image_path = os.path.join(uploads_dir, image_file_name)
            image_file.save(image_path)
            image_path = os.path.join(
                "static", os.path.join("uploads", image_file_name)
            )
            image_info["is_uploaded"] = True
            image_info["image_path"] = image_path
            image_info["image_file_name"] = image_file_name
            return render_template(
                "predict.html",
                caption="Click Generate button to generate caption",
                image_path=image_path,
                file_name=image_file_name,
            )
    return render_template(
        "index.html",
        caption="Upload Image. :)",
        image_path=os.path.join("static", "image_placeholder.png"),
        file_name="image_placeholder.png",
    )


@app.route("/predict", methods=["GET"])
def predict():
    if image_info["is_uploaded"] == False:
        image_info["is_uploaded"] = False
        return redirect(url_for("index"))

    image_path = image_info["image_path"]
    image_file_name = image_info["image_file_name"]
    # caption = get_prediction(predict_pipeline_obj, image_file_name)
    caption = "some dummy caption"
    caption = caption.capitalize()

    return render_template(
        "predict.html",
        caption=caption,
        image_path=image_path,
        file_name=image_file_name,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
