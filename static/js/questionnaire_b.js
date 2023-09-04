document.addEventListener('DOMContentLoaded', function() {
  const images = ['/static/img/graphs/1/11/three_portfolios.png', '/static/img/graphs/1/11/Gini_all_options.png'];
  let currentImageIndex = 0;

  const prevButton = document.querySelector('.carousel-prev');
  const nextButton = document.querySelector('.carousel-next');
  const prevButtonSvg = document.querySelector('.carousel-prev svg');
  const nextButtonSvg = document.querySelector('.carousel-next svg');

  // Initialize values
  prevButton.disabled = true; // Initialized value is disabled
  prevButtonSvg.classList.add('capital-market-disabled-button');
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
    const imgElement = document.getElementById('capital-market-img');

    imgElement.src = images[currentImageIndex];

    prevButton.disabled = currentImageIndex === 0;
    nextButton.disabled = currentImageIndex === images.length - 1;

    // Add or remove the opacity style based on the disabled state
    if (prevButton.disabled) {
      prevButtonSvg.classList.add('capital-market-disabled-button');
      prevButtonSvg.classList.remove('switch-img');
    } else {
      prevButtonSvg.classList.remove('capital-market-disabled-button');
      prevButtonSvg.classList.add('switch-img');
    }

    if (nextButton.disabled) {
      nextButtonSvg.classList.add('capital-market-disabled-button');
      nextButtonSvg.classList.remove('switch-img');
    } else {
      nextButtonSvg.classList.remove('capital-market-disabled-button');
      nextButtonSvg.classList.add('switch-img');
    }
  }
});
