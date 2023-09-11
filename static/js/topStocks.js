document.addEventListener('DOMContentLoaded', function() {
    function convertToTitleCase(str) {
        return str
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    function initializeCarousel(elementId) {
        const convertedString = convertToTitleCase(elementId);
        const imgElement = document.getElementById(`${elementId}-img`);
        const prevButton = document.querySelector(`.${elementId}-prev`);
        const nextButton = document.querySelector(`.${elementId}-next`);
        const prevButtonSvg = prevButton.querySelector('svg.switch-img');
        const nextButtonSvg = nextButton.querySelector('svg.switch-img');

        // Construct the image URLs using the working directory and Django template tags
        const baseImage = `/static/img/research/Top Stocks - ${convertedString}`;

        const images = [
            `${baseImage} (Graphs).png`, `${baseImage} (Table).png`,
        ];
        let currentImageIndex = 0;

        // Initialize values
        prevButton.disabled = true; // Initialized value is disabled
        prevButtonSvg.classList.add(`disabled-button`);
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
                prevButtonSvg.classList.add(`disabled-button`);
                prevButtonSvg.classList.remove('switch-img');
            } else {
                prevButtonSvg.classList.remove(`disabled-button`);
                prevButtonSvg.classList.add('switch-img');
            }

            if (nextButton.disabled) {
                nextButtonSvg.classList.add(`disabled-button`);
                nextButtonSvg.classList.remove('switch-img');
            } else {
                nextButtonSvg.classList.remove(`disabled-button`);
                nextButtonSvg.classList.add('switch-img');
            }
        }

        // Initial image display
        updateImageDisplay();
    }

    // Call the function for each element you want to target
    const classesArray = [
        'israel-stocks-indexes', 'israel-general-bonds-indexes', 'israel-government-bonds-indexes', 'us-stocks-indexes',
        'us-bonds-indexes', 'us-commodity-indexes', 'us-stocks', 'israel-stocks', 'all',
    ];
    classesArray.forEach((className) => {
        initializeCarousel(className);
    });
});