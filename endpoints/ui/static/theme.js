// Picks the page theme before the first paint (loaded synchronously in <head>,
// ahead of the stylesheets' first use), so a dark theme doesn't flash white.
// app.js builds the theme menu from this list and calls apply() on changes.

(() => {
  const THEMES = [
    { id: "", label: "System" },
    { id: "glass", label: "Glass" },
    { id: "glass-dark", label: "Glass (dark)" },
    { id: "gippity", label: "Gippity" },
    { id: "gippity-dark", label: "Gippity (dark)" },
    { id: "ember", label: "Ember" },
    { id: "unicorn", label: "Unicorn" },
    { id: "unicorn-dark", label: "Unicorn (dark)" },
    { id: "matrix", label: "Matrix" },
    { id: "hot-dog-stand", label: "Hot dog stand" },
  ];
  // "System" follows the OS light/dark preference with these two
  const SYSTEM_LIGHT = "glass";
  const SYSTEM_DARK = "glass-dark";

  // Names stored by earlier versions of the client
  const RENAMED = { light: "gippity", dark: "gippity-dark", modern: "glass-dark" };

  const prefersDark = matchMedia("(prefers-color-scheme: dark)");
  let choice = "";

  function normalize(name) {
    name = RENAMED[name] ?? name ?? "";
    return THEMES.some((t) => t.id === name) ? name : "";
  }

  function resolve(name) {
    return normalize(name) || (prefersDark.matches ? SYSTEM_DARK : SYSTEM_LIGHT);
  }

  function apply(name) {
    choice = normalize(name);
    const theme = resolve(choice);
    if (document.documentElement.dataset.theme !== theme) {
      document.documentElement.dataset.theme = theme;
      document.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
    }
  }

  prefersDark.addEventListener("change", () => apply(choice));

  try {
    apply(JSON.parse(localStorage.getItem("tabby-ui.settings") || "{}").theme);
  } catch {
    apply("");
  }

  window.tabbyTheme = { THEMES, normalize, resolve, apply };
})();
