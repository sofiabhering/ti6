// script.js
document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var imageInput = document.getElementById('image-input');
    var resultContainer = document.getElementById('result-container');
    resultContainer.classList.add('hidden');

    if (imageInput.files.length === 0) {
        alert('Please select an image to upload.');
        return;
    }

    var file = imageInput.files[0];
    var formData = new FormData();
    formData.append('image', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            var result = JSON.parse(xhr.responseText);
            document.getElementById('result').textContent = 'Age: ' + result.age;
            resultContainer.classList.remove('hidden');
        } else {
            alert('Error: ' + xhr.statusText);
        }
    };
    xhr.onerror = function() {
        alert('Error: ' + xhr.statusText);
    };
    xhr.send(formData);
});
