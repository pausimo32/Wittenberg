/**
 * app.js
 * Wittenberg — Logique partagée de l'application
 * Navigation, composants communs, utilitaires UI
 */

// ── Navigation active ─────────────────────────────────────────────────────
(function highlightActiveNav() {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link, .nav-mobile-menu a').forEach(link => {
    const href = link.getAttribute('href') || '';
    const linkPage = href.split('/').pop();
    if (linkPage === path || (path === 'index.html' && linkPage === '')) {
      link.classList.add('active');
    }
  });
})();

// ── Mobile menu toggle ───────────────────────────────────────────────────
(function mobileMenu() {
  const toggle = document.getElementById('nav-toggle');
  const menu   = document.getElementById('nav-mobile-menu');
  if (!toggle || !menu) return;

  toggle.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open);
    document.body.style.overflow = open ? 'hidden' : '';
  });

  // Fermer si on clique sur un lien
  menu.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      menu.classList.remove('open');
      document.body.style.overflow = '';
    });
  });
})();

// ── Scroll animations (IntersectionObserver) ────────────────────────────
(function scrollAnimations() {
  const els = document.querySelectorAll('.slide-up');
  if (!els.length) return;

  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  els.forEach(el => observer.observe(el));
})();

// ── Tabs ──────────────────────────────────────────────────────────────────
(function initTabs() {
  document.querySelectorAll('.tabs').forEach(tabGroup => {
    const tabs   = tabGroup.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;

        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        panels.forEach(p => {
          p.classList.toggle('active', p.dataset.panel === target);
        });
      });
    });

    // Activer le premier onglet par défaut
    if (tabs.length > 0 && !tabGroup.querySelector('.tab.active')) {
      tabs[0].click();
    }
  });
})();

// ── Utilitaires ───────────────────────────────────────────────────────────
function formatFrequency(hz) {
  return hz >= 1000
    ? (hz / 1000).toFixed(hz % 1000 === 0 ? 0 : 1).replace('.', ',') + ' kHz'
    : hz + ' Hz';
}

function formatConfidence(ratio) {
  return Math.round(ratio * 100) + ' %';
}

function getConfidenceLabel(ratio) {
  var i = window.WI18n;
  if (ratio >= 0.85) return { text: i ? i.t('analyser.conf_high')   : 'Très élevée', cls: 'badge-success' };
  if (ratio >= 0.70) return { text: i ? i.t('analyser.conf_med_hi') : 'Élevée',      cls: 'badge-primary' };
  if (ratio >= 0.55) return { text: i ? i.t('analyser.conf_med')    : 'Moyenne',     cls: 'badge-warn' };
  return                     { text: i ? i.t('analyser.conf_low')    : 'Faible',      cls: 'badge-muted' };
}

// ── Chladni Canvas (décoration animée) ───────────────────────────────────
function drawChladniPattern(canvas, m, n, options = {}) {
  const ctx  = canvas.getContext('2d');
  const W    = canvas.width;
  const H    = canvas.height;
  const step = options.step || 2;
  const color = options.color || '#205f75';
  const threshold = options.threshold || 0.04;

  ctx.clearRect(0, 0, W, H);

  for (let px = 0; px < W; px += step) {
    for (let py = 0; py < H; py += step) {
      const x = (px / W) * Math.PI;
      const y = (py / H) * Math.PI;
      const z = Math.cos(m * x) * Math.cos(n * y) - Math.cos(n * x) * Math.cos(m * y);

      if (Math.abs(z) < threshold) {
        ctx.fillStyle = color;
        ctx.fillRect(px, py, step, step);
      }
    }
  }
}

// Initialiser les canvas décoratifs présents sur la page
(function initChladniCanvases() {
  document.querySelectorAll('canvas[data-chladni]').forEach(canvas => {
    const m = parseInt(canvas.dataset.m || '2', 10);
    const n = parseInt(canvas.dataset.n || '3', 10);
    const color = canvas.dataset.color || '#205f75';
    drawChladniPattern(canvas, m, n, { color, step: 2, threshold: 0.05 });
  });
})();

// ── i18n init ────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  if (window.WI18n) window.WI18n.init();
});
