document.addEventListener('DOMContentLoaded', (event) => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('outputCanvas');
    const ctx = canvas.getContext('2d');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
            video.play();
        })
        .catch((error) => {
            console.error('Error accessing webcam:', error);
        });

    video.addEventListener('loadeddata', () => {
        setInterval(() => {
            captureAndSendFrame();
        }, 500); // Adjust the interval as needed
    });

    function captureAndSendFrame() {
       
        const imageData = canvas.toDataURL('image/jpeg', 1); // Convert canvas to base64 data

        // Send imageData to the Python server
        sendDataToServer(imageData);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    function sendDataToServer(imageData) {
        fetch('/predictions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_base64: imageData }),
        })
        .then(response => response.json())
        .then((data) => {
            // Process the results and draw bounding boxes on the canvas
            console.log(data)
            drawBoundingBoxes(data);
        })
        .catch((error) => {
            console.error('Error sending data to server:', error);
        });
    }

    function drawBoundingBoxes(result) {
        // Clear previous bounding boxes
        const boundingBoxContainer = document.getElementById('boundingBoxContainer');
        boundingBoxContainer.innerHTML = '';
        result.data.forEach(result => {
            const boundingBox = document.createElement('div');
            const width = Math.floor((result.box[2]));
            const height = Math.floor((result.box[3]));
            const x = Math.floor(result.box[0]);
            const y = Math.floor(result.box[1]);
            console.log(width, height, x, y);
            boundingBox.className = 'bounding-box';
            boundingBox.style.position = 'absolute';
            boundingBox.style.left = `${x}px`;
            boundingBox.style.top = `${y}px`;
            boundingBox.style.width = `${width}px`;
            boundingBox.style.height = `${height}px`;
            const label = document.createElement('span');
            label.className = 'label';
            label.innerText = result.class_name;

            boundingBox.appendChild(label);
            boundingBoxContainer.appendChild(boundingBox);
        });
    }
});
