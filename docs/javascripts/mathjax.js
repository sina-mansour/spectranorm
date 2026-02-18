window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // We remove the broad "ignore" and specifically target the Jupyter classes
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown"
  }
};

document.addEventListener("DOMContentLoaded", function() {
  if (typeof MathJax !== 'undefined') {
    MathJax.typesetPromise();
  }
});

// This handles cases where the notebook content is loaded dynamically
if (typeof IPython !== 'undefined') {
    MathJax.typesetPromise();
}
