// Entry point: lazy page modules, the mode label, a subtle random-walk background, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'analyze.js': ['analyze'], 'tracks.js': ['tracks'], 'plots.js': ['plots'],
  'validate.js': ['validate'], 'docs.js': ['method', 'manual', 'about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: demo data analysed at build time by the Python code; simple-method uploads run in your browser'
  : 'local app: analysis runs in Python on this computer';

// Background: a few particles doing 2D random walks, leaving faint trails. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; g.fillStyle = '#05060c'; g.fillRect(0, 0, canvas.width, canvas.height); };
  fit(); addEventListener('resize', fit);
  const cols = ['#2ee6ff', '#ff4fd8', '#ffb02e'];
  const ps = Array.from({ length: 14 }, (_, i) => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight,
    c: cols[i % 3], d: i % 3 === 0 ? Math.random() * 2 * Math.PI : null }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 60) return;
    last = t;
    g.fillStyle = 'rgba(5,6,12,.035)'; g.fillRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      const nx = p.x + (Math.random() - 0.5) * 6 + (p.d === null ? 0 : Math.cos(p.d) * 1.5);
      const ny = p.y + (Math.random() - 0.5) * 6 + (p.d === null ? 0 : Math.sin(p.d) * 1.5);
      g.strokeStyle = p.c; g.lineWidth = 1.2; g.beginPath(); g.moveTo(p.x, p.y); g.lineTo(nx, ny); g.stroke();
      p.x = (nx + canvas.width) % canvas.width; p.y = (ny + canvas.height) % canvas.height;
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
