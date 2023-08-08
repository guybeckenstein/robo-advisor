document.addEventListener('DOMContentLoaded', function() {
  const pageHeight = document.body.scrollHeight;
  document.documentElement.style.setProperty('--page-height', `${pageHeight}px`);

  // Check page height and adjust footer position if needed
  if (pageHeight < 1080) {
    const footer = document.querySelector('footer');
    footer.style.bottom = '0';
    footer.style.top = 'auto';
  }
});