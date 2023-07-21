import os, cv2, argparse, easyocr
from flask import Flask, request, jsonify, send_file
import numpy as np
from transformers import pipeline


app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'image' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    image = request.files['image']

    # Save the file temporarily
    file_path = 'temp.jpg'
    image.save(file_path)

    # Perform OCR on the image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(file_path, detail=0)

    # Delete the temporary file
    os.remove(file_path)

    # Return the OCR result
    return jsonify({'result': '\n'.join(result)})



@app.route('/review', methods=['POST'])
def review():
    sentiment_pipeline = pipeline("sentiment-analysis")
    return sentiment_pipeline(request.form['review'])



@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No file found'}), 400
    # Get the uploaded image from the request
    image_file = request.files['image']

    args = argparse.Namespace()
    args.config = 'yolov3.cfg'
    args.weights = 'yolov3.weights'
    args.classes = 'yolov3.txt'

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)
    
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label + ' ' + str(round(confidence, 2)), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Perform object detection
    image = cv2.imread(temp_image_path)
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    net = cv2.dnn.readNet(args.weights, args.config)
    
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]
        
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    output_image_path = 'output_image.jpg'
    cv2.imwrite(output_image_path, image)

    return send_file(output_image_path, mimetype='image/jpeg')

@app.route('/label', methods=['POST'])
def label():
    if 'image' not in request.files:
        return 'No image file uploaded', 400
    # Get the uploaded image from the request
    image_file = request.files['image']

    args = argparse.Namespace()
    args.config = 'yolov3.cfg'
    args.weights = 'yolov3.weights'
    args.classes = 'yolov3.txt'

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # Save the image to a temporary file
    temp_image_path = 'temp_image.jpg'
    image_file.save(temp_image_path)
    

    # Perform object detection
    image = cv2.imread(temp_image_path)
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    net = cv2.dnn.readNet(args.weights, args.config)
    
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    

    detected_labels = []
    for i in indices:
        label = classes[class_ids[i]]
        detected_labels.append(label)

    response = {
        'Label': detected_labels
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=7777)
