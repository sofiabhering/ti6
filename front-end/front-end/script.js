document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var imageInput = document.getElementById('image-input');
    var resultContainer = document.getElementById('result-container');
    // resultContainer.classList.add('hidden');

    if (imageInput.files.length === 0) {
        alert('Selecione uma imagem');
        return;
    }

    var file = imageInput.files[0];
    var formData = new FormData();
    formData.append('image', file);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
        mode: 'cors',
        headers: {
            'Access-Control-Allow-Origin': '*', // Required for CORS support to work 
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Response Error: ' + response.statusText);
        }
        return response.json();
    })
    .then(result => {
        document.getElementById('result').textContent = 'Age: ' + result.age;
        // resultContainer.classList.remove('hidden');
    })
    .catch(error => {
        alert('Error: ' + error.message);
    });
});

