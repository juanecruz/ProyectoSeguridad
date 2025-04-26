document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predictBtn');
    const clearBtn = document.getElementById('clearBtn');
    const attackBtn = document.getElementById('attackBtn');
    const predictionElement = document.getElementById('prediction');
    const confidenceElement = document.getElementById('confidence');

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(
            e.type === 'touchstart' ? 'mousedown' : 'mousemove',
            {
                clientX: touch.clientX,
                clientY: touch.clientY
            }
        );
        canvas.dispatchEvent(mouseEvent);
    }

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    clearBtn.addEventListener('click', function() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionElement.textContent = '-';
        confidenceElement.textContent = '-';
    });

    predictBtn.addEventListener('click', function() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);

        const imageData = tempCanvas.toDataURL('image/png');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);

            predictionElement.textContent = data.prediction;
            confidenceElement.textContent = (data.confidence * 100).toFixed(1) + '%';
        })
        .catch(error => {
            predictionElement.textContent = 'Error';
            confidenceElement.textContent = error.message;
        });
    });

    attackBtn.addEventListener('click', function() {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);

        const imageData = tempCanvas.toDataURL('image/png');

        fetch('/attack', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            predictionElement.textContent = data.prediction;
            confidenceElement.textContent = (data.confidence * 100).toFixed(1) + '%';

            const img = new Image();
            img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            img.src = data.adversarial_image;
        })
        .catch(error => {
            predictionElement.textContent = 'Error';
            confidenceElement.textContent = error.message;
        });
    });

    const defenseBtn = document.getElementById('defenseBtn');

defenseBtn.addEventListener('click', function() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCanvas.toDataURL('image/png');

    fetch('/defend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        predictionElement.textContent = data.prediction;
        confidenceElement.textContent = (data.confidence * 100).toFixed(1) + '%';

        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        img.src = data.cleaned_image;
    })
    .catch(error => {
        predictionElement.textContent = 'Error';
        confidenceElement.textContent = error.message;
    });
});

});
