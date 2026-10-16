// Decorations that some themes add on top of their CSS. Each one runs only
// while its theme is active and is torn down when the theme changes, so the
// other themes pay nothing for them.

const EFFECTS = {
  matrix: [telemetry],
};

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let stopActive = [];

function update(theme) {
  for (const stop of stopActive) stop();
  stopActive = (EFFECTS[theme] || []).map((start) => start()).filter(Boolean);
}

// Matrix: a readout of entirely meaningless numbers in the top bar

function telemetry() {
  const topbar = document.getElementById("topbar");
  const hud = document.createElement("div");
  hud.className = "theme-hud";
  hud.setAttribute("aria-hidden", "true");
  topbar.append(hud);

  const hex = (digits) => ((Math.random() * 16 ** digits) | 0).toString(16).toUpperCase().padStart(digits, "0");
  let sync = 97.3;
  function tick() {
    sync = Math.min(99.99, Math.max(91, sync + (Math.random() - 0.48) * 0.8));
    hud.textContent =
      `NODE ${hex(2)}:${hex(2)} │ SYNC ${sync.toFixed(2)}% │ ` +
      `ENTROPY 0.${hex(4)} │ IRQ ${Math.random() < 0.04 ? "MEOW" : "OK"}`;
  }
  tick();
  const timer = setInterval(tick, 700);

  return () => {
    clearInterval(timer);
    hud.remove();
  };
}

document.addEventListener("themechange", (e) => update(e.detail));
reducedMotion.addEventListener("change", () => update(document.documentElement.dataset.theme));
update(document.documentElement.dataset.theme);
