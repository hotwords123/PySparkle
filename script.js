document.addEventListener('DOMContentLoaded', function () {
  const container = document.querySelector('div.highlight');

  let callback = null;

  container.addEventListener('click', function (evt) {
    /** @type {Element} */
    const el = evt.target;

    if (el.matches('span.token[data-target]')) {
      const targetEl = document.getElementById(el.getAttribute('data-target'));
      if (targetEl) {
        targetEl.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
          inline: 'center'
        });

        callback && callback();

        targetEl.classList.add('focus');
        const timer = setTimeout(callback = () => {
          targetEl.classList.remove('focus');
          callback = null;
          clearTimeout(timer);
        }, 1000);
      }
    }
  });
});
