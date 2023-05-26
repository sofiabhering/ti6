document
  .getElementById("upload-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    var files = document.getElementById("file-input").files;
    var formData = new FormData();

    for (var i = 0; i < files.length; i++) {
      var file = files[i];
      formData.append("images[]", file, file.name);
    }

    var request = new XMLHttpRequest();
    request.open("POST", "http://127.0.0.1:5000/predict");

    request.onload = function () {
      if (request.status === 200) {
        var response = JSON.parse(request.responseText);
        var processedImages = response.processed_images;
        var imageGallery = document.getElementById("image-gallery");
        
        // Clear previous images
        imageGallery.innerHTML = "";
    
        for (var i = 0; i < processedImages.length; i++) {
          var processedImage = processedImages[i];
          var imageElement = document.createElement("img");
          imageElement.src = "uploads/" + processedImage;
          imageGallery.appendChild(imageElement);
        }
    
        resultContainer.classList.remove("hidden");
      } else {
        alert("Houve um erro ao enviar as imagens.");
      }
    };

    request.send(formData);
  });

document.getElementById("file-input").addEventListener("change", function () {
  var files = Array.from(this.files);
  var selectedFilesContainer = document.getElementById("selected-files");

  selectedFilesContainer.innerHTML = "";

  files.forEach(function (file) {
    var reader = new FileReader();

    reader.onload = function (event) {
      var img = document.createElement("img");
      img.src = event.target.result;
      selectedFilesContainer.appendChild(img);
    };

    reader.readAsDataURL(file);
  });
});

document
  .getElementById("download-button")
  .addEventListener("click", function () {
    var links = document.querySelectorAll("#download-links a");
    for (var i = 0; i < links.length; i++) {
      links[i].click();
    }
  });
