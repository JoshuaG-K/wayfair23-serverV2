document.addEventListener('DOMContentLoaded', function() {
    var items = document.querySelectorAll('.item');
    items.forEach(function(item) {
        var image = item.querySelector('.preview-image');
        var video = item.querySelector('.video');
        var timeoutId;

        item.addEventListener('mouseover', function() {
            timeoutId = setTimeout(function() {
                image.style.display = 'none';
                video.style.display = 'block';
                video.play();
            }, 500); // Delay of 0.5 seconds
        });

        item.addEventListener('mouseout', function() {
            clearTimeout(timeoutId);
            video.pause();
            video.currentTime = 0;
            video.style.display = 'none';
            image.style.display = 'block';
        });
    });
});
