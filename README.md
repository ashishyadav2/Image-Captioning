# Image Caption Generator
![background_image](assets/bg.png)

## Project Details (**Use git branch `V2`**)
#### Overview:
This project revolves around creating an **Image Caption Generator** that takes an image as input and outputs a descriptive sentence in natural language. The generator combines advanced deep learning techniques to understand and describe the content of an image.
- Built an image caption generator using Flask and an LSTM model.
- Utilized a pre-trained image recognition model (e.g., VGG16) to extract image features for the LSTM model.
- Users can upload images and receive a natural language description.
- Applications include assisting visually impaired users and getting scene description.

#### Key Components:

1. **Flask Framework**:
   - Flask is used to create a web interface where users can upload images.
   - It provides a lightweight and efficient way to handle HTTP requests and serve the application.

2. **LSTM (Long Short-Term Memory) Model**:
   - LSTM is a type of recurrent neural network (RNN) designed to process sequences, making it suitable for generating text descriptions.
   - In this project, the LSTM model takes image features and generates coherent, natural language captions.

3. **VGG16 Model for Feature Extraction**:
   - VGG16 is a deep convolutional neural network pre-trained on a large dataset (ImageNet) to recognize a wide variety of images.
   - For this project, VGG16 is used to extract high-level features from images, which serve as input to the LSTM model.

#### Working:

1. **Image Upload**:
   - Users can upload an image through the Flask web interface.

2. **Feature Extraction**:
   - The uploaded image is passed through the VGG16 model to extract relevant features.
   - These features encapsulate the important visual elements of the image.

3. **Caption Generation**:
   - The extracted features are fed into the LSTM model, which processes them to generate a textual description.
   - The output is a complete sentence that describes the contents of the image.

4. **Output**:
   - The generated caption is displayed to the user, providing an accurate and meaningful description of the image.

#### Applications:

1. **Assisting Visually Impaired Users**:
   - The tool can provide descriptions of images, helping visually impaired individuals understand visual content.
   
2. **Scene Description**:
   - It can be used in applications requiring automatic scene analysis and description, such as surveillance or content categorization.

3. **Content Tagging and Management**:
   - Useful for automatically tagging and managing large collections of images, making it easier to organize and search through visual data.

#### Docker Integration:

To enhance the deployment and scalability of this project, Docker is used to containerize the entire application. Docker ensures that the application runs consistently across different environments by encapsulating all dependencies and configurations. This makes it easier to deploy, update, and scale the application efficiently.

## RUN
1. Download [Docker](https://www.docker.com/products/docker-desktop/) for Desktop
2. Open `CMD Prompt` or any terminal
3. RUN `docker pull hydrogencyanide/image-caption-gen`
4. RUN `docker run -p 9000:5000 hydrogencyanide/image-caption-gen`
5. Open any `browser` and goto `http://127.0.0.1:9000/`
6. To stop docker container:
    * Open `CMD Prompt` in a new window
    * `docker ps`:  To get all active container information along container ID
    * `docker stop container_id` : Use corresponding container ID to stop container