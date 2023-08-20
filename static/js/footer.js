document.addEventListener('DOMContentLoaded', function() {
  const pageHeight = document.body.scrollHeight;
  document.documentElement.style.setProperty('--page-height', `${pageHeight}px`);

  // Check if footer element exists
  const footer = document.querySelector('footer');
  if (footer) {
    // Check page height and adjust footer position if needed
    if (pageHeight < 1080) {
      footer.style.bottom = '0';
      footer.style.top = 'auto';
    }
  }
});