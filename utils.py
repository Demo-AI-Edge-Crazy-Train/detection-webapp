import cv2.dnn
import numpy as np
import base64
import time

CLASSES = {
    0: "SpeedLimit",
    1: "DangerAhead"
}
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def preprocess(image_b64):
    image_b64 = image_b64.split("data:image/jpeg;base64,")[1]
    img = base64.b64decode(image_b64)
    imgar = np.frombuffer(img, dtype=np.uint8)
    original_image = cv2.imdecode(imgar, cv2.IMREAD_COLOR)
    #original_image: np.ndarray = cv2.imdecode(img, cv2.IMREAD_COLOR)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    return blob, scale, original_image


def postprocess(response, scale, original_image, CONF_THRESHOLD):
    outputs = np.array([cv2.transpose(response[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        if scores[index] >= CONF_THRESHOLD:
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': [str(b) for b in box],
                'scale': str(scale)}
            detections.append(detection)
        # draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
        #                   round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
    return detections


def send_request(image, ort_sess):
    outputs = ort_sess.run(None, {'images': image})
    return outputs


def process_image(image_b64, ort_sess, CONF_THRESHOLD):
    start_pre = time.time()
    preprocessed, scale, original_image = preprocess(image_b64)
    time_pre = time.time() - start_pre
    start_inf = time.time()
    response = send_request(preprocessed, ort_sess)
    time_inf = time.time() - start_inf
    start_post = time.time()
    detections = postprocess(response[0], scale, original_image, CONF_THRESHOLD)
    time_post = time.time() - start_post
    print(f"pre: {time_pre}, inf: {time_inf}, post: {time_post}")

    return detections