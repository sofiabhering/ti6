document
  .getElementById("upload-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();
  });

document
  .getElementById("send-button")
  .addEventListener("click", function (event) {
    var files = document.getElementById("file-input").files;
    var formData = new FormData();

    for (var i = 0; i < files.length; i++) {
      var file = files[i];
      formData.append("images[]", file, file.name);
    }

    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
      // mode: "cors",
      // headers: {
      //   "Access-Control-Allow-Origin": "*", // Required for CORS support to work
      // },
  })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Response Error: " + response.statusText);
        }
        console.log("dsjfsaçjdfksadklçfkfsadj")
        return response.json();
      })
      .then(function (data) {
        console.log(data)
        console.log("Processadas !!")
        var processedImages = data.processed_images;
        var imageGallery = document.getElementById("image-gallery");
        var downloadLinksContainer = document.getElementById("download-links");

        // Clear previous images and download links
        imageGallery.innerHTML = "";
        downloadLinksContainer.innerHTML = "";

        for (var i = 0; i < processedImages.length; i++) {
          var processedImage = processedImages[i];
          // var imageElement = document.createElement("img");
          // imageElement.src = "uploads/" + processedImage;
          // imageGallery.appendChild(imageElement);

          // Create a download link for each processed image
          var downloadLink = document.createElement("a");
          downloadLink.href = "uploads/" + processedImage;
          downloadLink.download = processedImage;
          downloadLink.innerText = "Download Image " + (i + 1);
          downloadLinksContainer.appendChild(downloadLink);
        }

        document.getElementById("result-container").classList.remove("hidden");
      })
      .catch(function (e) {
        throw new Error("Error" + e);
      });
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
