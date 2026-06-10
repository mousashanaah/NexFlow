/* Custom Balls page — two-line live studio, colour, FAQ accordion */
(() => {
  "use strict";
  const lines = document.querySelectorAll("#ballLines [data-line]");
  const l1 = document.getElementById("line1");
  const l2 = document.getElementById("line2");

  const sync = () => {
    if (lines[0]) lines[0].textContent = (l1.value.trim() || "YOUR").toUpperCase();
    if (lines[1]) lines[1].textContent = (l2.value.trim() || "NAME").toUpperCase();
  };
  l1?.addEventListener("input", sync);
  l2?.addEventListener("input", sync);
  sync();

  document.getElementById("swatches")?.addEventListener("click", (e) => {
    const sw = e.target.closest(".swatch");
    if (!sw) return;
    document.querySelectorAll("#swatches .swatch").forEach((s) => s.classList.remove("active"));
    sw.classList.add("active");
    lines.forEach((b) => (b.style.color = sw.dataset.color));
  });

  document.getElementById("logoDrop")?.addEventListener("click", function () {
    this.textContent = "✓  Great — attach your logo when you order or via WhatsApp.";
    this.style.color = "#7ac72e";
    this.style.borderColor = "#7ac72e";
  });

  /* FAQ accordion */
  document.getElementById("faq")?.addEventListener("click", (e) => {
    const q = e.target.closest(".faq-q");
    if (!q) return;
    const item = q.parentElement;
    const wasOpen = item.classList.contains("open");
    document.querySelectorAll("#faq .faq-item").forEach((i) => i.classList.remove("open"));
    if (!wasOpen) item.classList.add("open");
  });

  /* Bulk form */
  document.getElementById("bulkForm")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    const o = btn.innerHTML;
    btn.innerHTML = "✓ Quote requested!";
    btn.style.background = "var(--green-deep)";
    btn.style.color = "#fff";
    setTimeout(() => { btn.innerHTML = o; btn.style.background = ""; btn.style.color = ""; e.target.reset(); }, 2200);
  });
})();
