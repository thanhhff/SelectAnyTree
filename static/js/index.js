/* Smooth-scroll for anchor links */
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

/* Restart GIFs from frame 1 the moment they enter the viewport.
   This eliminates the perceived delay caused by the browser resuming
   a mid-animation GIF that started downloading before it was visible. */
(function () {
  const gifs = document.querySelectorAll('.demo-gif-wrap img');
  if (!gifs.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        const src = img.src;
        img.src = '';          // force browser to reset the animation
        img.src = src;
        observer.unobserve(img); // only restart once per page visit
      }
    });
  }, {
    threshold: 0.15           // trigger when 15% of the image is visible
  });

  gifs.forEach(img => {
    // If already loaded and visible on page load (e.g. large screen), restart now
    if (img.complete && img.getBoundingClientRect().top < window.innerHeight) {
      const src = img.src;
      img.src = '';
      img.src = src;
    } else {
      observer.observe(img);
    }
  });
})();
