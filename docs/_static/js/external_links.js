document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('a[href^="http"], a.external').forEach(function(link) {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');  // Security best practice
  });
});
