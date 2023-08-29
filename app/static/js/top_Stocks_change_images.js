document.addEventListener('DOMContentLoaded', function() {
    const image = document.getElementById('capital-market-swapped');
    if (image) {
        const images = [
        'C:/src/github.com/guybeckenstein/robo-advisor/static/img/research/Top Stocks All intersection.png',
        'C:/src/github.com/guybeckenstein/robo-advisor/static/img/research/top_stocks_US stocks.png'
        ];
        let imageId = 0;

        setInterval(function() {
            image.src = images[imageId % images.length];
            imageId++;
        }, 800);
    }
});