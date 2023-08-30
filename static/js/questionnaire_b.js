document.addEventListener('DOMContentLoaded', function() {
    const image = document.getElementById('capital-market-swapped');
    if (image) {
        const images = [
            'C:/src/github.com/guybeckenstein/robo-advisor/static/img/graphs/1/11/all_options.png',
            'C:/src/github.com/guybeckenstein/robo-advisor/static/img/graphs/1/11/three_portfolios.png'
        ];
        let imageId = 0;

        setInterval(function() {
            image.src = images[imageId % images.length];
            imageId++;
        }, 800);
    }
});