document.addEventListener('DOMContentLoaded', function() {
  const imgElement = document.getElementById('capital-market-img');
  const prevButton = document.querySelector('.carousel.carousel-prev');
  const nextButton = document.querySelector('.carousel.carousel-next');
  const prevButtonSvg = prevButton.querySelector('svg.switch-img');
  const nextButtonSvg = nextButton.querySelector('svg.switch-img');
  const images = [imgElement.getAttribute('data-second-graph'), imgElement.getAttribute('data-third-graph')];
  let currentImageIndex = 0;

  // Initialize values
  prevButton.disabled = true; // Initialized value is disabled
  prevButtonSvg.classList.add('disabled-button');
  prevButtonSvg.classList.remove('switch-img');

  prevButton.addEventListener('click', () => {
    if (currentImageIndex > 0) {
      currentImageIndex--;
      updateImageDisplay();
    }
  });

  nextButton.addEventListener('click', () => {
    if (currentImageIndex < images.length - 1) {
      currentImageIndex++;
      updateImageDisplay();
    }
  });

  function updateImageDisplay() {
    imgElement.src = images[currentImageIndex];

    prevButton.disabled = currentImageIndex === 0;
    nextButton.disabled = currentImageIndex === images.length - 1;

    // Add or remove the opacity style based on the disabled state
    if (prevButton.disabled) {
      prevButtonSvg.classList.add('disabled-button');
      prevButtonSvg.classList.remove('switch-img');
    } else {
      prevButtonSvg.classList.remove('disabled-button');
      prevButtonSvg.classList.add('switch-img');
    }

    if (nextButton.disabled) {
      nextButtonSvg.classList.add('disabled-button');
      nextButtonSvg.classList.remove('switch-img');
    } else {
      nextButtonSvg.classList.remove('disabled-button');
      nextButtonSvg.classList.add('switch-img');
    }
  }

  // Initial image display
  updateImageDisplay();
});

