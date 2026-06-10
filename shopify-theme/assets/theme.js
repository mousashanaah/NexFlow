/* =================================================================
   GOLF4LESS — Shopify theme interactions
   ================================================================= */
(() => {
  "use strict";
  const $ = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => [...c.querySelectorAll(s)];

  /* ---------- Header shadow ---------- */
  const header = $("#header");
  if (header) {
    const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 10);
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
  }

  /* ---------- Mobile drawer ---------- */
  const drawer = $("#drawer");
  $("#burger")?.addEventListener("click", () => drawer.classList.add("open"));
  drawer?.addEventListener("click", (e) => { if (e.target.matches("[data-close]")) drawer.classList.remove("open"); });

  /* ---------- Reveal on scroll ---------- */
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }), { threshold: 0.14 });
  $$(".reveal").forEach((el) => io.observe(el));

  /* ---------- Animated counters ---------- */
  const cio = new IntersectionObserver((es) => es.forEach((e) => {
    if (!e.isIntersecting) return;
    const el = e.target, target = parseFloat(el.dataset.count) || 0;
    let cur = 0; const step = Math.max(1, target / 60);
    const run = () => { cur += step; if (cur >= target) el.textContent = target.toLocaleString(); else { el.textContent = Math.floor(cur).toLocaleString(); requestAnimationFrame(run); } };
    run(); cio.unobserve(el);
  }), { threshold: 0.5 });
  $$("[data-count]").forEach((c) => cio.observe(c));

  /* ---------- Daily deal countdown ---------- */
  const cd = $("[data-countdown]");
  if (cd) {
    const h = $("[data-h]", cd), m = $("[data-m]", cd), s = $("[data-s]", cd);
    const tick = () => {
      const now = new Date(), end = new Date(now); end.setHours(24, 0, 0, 0);
      let d = Math.floor((end - now) / 1000);
      h.textContent = String(Math.floor(d / 3600)).padStart(2, "0");
      m.textContent = String(Math.floor((d % 3600) / 60)).padStart(2, "0");
      s.textContent = String(d % 60).padStart(2, "0");
    };
    tick(); setInterval(tick, 1000);
  }

  /* ---------- Ball customiser ---------- */
  const ballInput = $("[data-ball-input]");
  if (ballInput) {
    const ballText = $("[data-ball-text]");
    ballInput.addEventListener("input", () => { ballText.textContent = (ballInput.value.trim() || "YOUR TEXT").toUpperCase(); });
    $("[data-swatches]")?.addEventListener("click", (e) => {
      const sw = e.target.closest(".swatch"); if (!sw) return;
      $$(".swatch").forEach((x) => x.classList.remove("active")); sw.classList.add("active");
      ballText.style.color = sw.dataset.color;
    });
  }

  /* ---------- Cart count helper ---------- */
  const updateCartCount = async () => {
    try {
      const r = await fetch("/cart.js"); const c = await r.json();
      $$("#cartCount").forEach((el) => { el.textContent = c.item_count; });
    } catch (e) {}
  };

  /* ---------- AJAX add-to-cart (cards + simple buttons) ---------- */
  document.addEventListener("submit", async (e) => {
    const form = e.target;
    if (!form.matches(".card-form")) return;
    e.preventDefault();
    const btn = $("button[type=submit]", form);
    const original = btn.innerHTML;
    btn.disabled = true;
    try {
      const fd = new FormData(form);
      const r = await fetch("/cart/add.js", { method: "POST", body: fd, headers: { "X-Requested-With": "XMLHttpRequest" } });
      if (!r.ok) throw new Error();
      btn.innerHTML = "✓ Added"; btn.style.background = "var(--green-deep)"; btn.style.color = "#fff";
      updateCartCount();
    } catch { btn.innerHTML = "Try again"; }
    setTimeout(() => { btn.innerHTML = original; btn.style.background = ""; btn.style.color = ""; btn.disabled = false; }, 1300);
  });

  /* ---------- PDP: gallery, options, qty ---------- */
  const mainImg = $("[data-main-img]");
  $("[data-thumbs]")?.addEventListener("click", (e) => {
    const t = e.target.closest(".thumb"); if (!t) return;
    $$(".thumb").forEach((x) => x.classList.remove("active")); t.classList.add("active");
    if (mainImg && t.dataset.img) mainImg.src = t.dataset.img;
  });

  const variantSelect = $("[data-variant-select]");
  const selected = [];
  $$("[data-option-index]").forEach((group) => {
    const idx = +group.dataset.optionIndex;
    const init = $(".opt.active", group);
    if (init) selected[idx] = init.dataset.value;
    group.addEventListener("click", (e) => {
      const o = e.target.closest(".opt"); if (!o) return;
      $$(".opt", group).forEach((x) => x.classList.remove("active")); o.classList.add("active");
      selected[idx] = o.dataset.value;
      // Match the variant by its title (Shopify joins option values with " / ")
      if (variantSelect) {
        const want = selected.join(" / ");
        $$("option", variantSelect).forEach((opt) => { if (opt.textContent.trim() === want) variantSelect.value = opt.value; });
      }
    });
  });

  const qVal = $("[data-q-val]"), qInput = $("[data-q-input]");
  let q = 1;
  $("[data-q-plus]")?.addEventListener("click", () => { q++; qVal.textContent = q; if (qInput) qInput.value = q; });
  $("[data-q-minus]")?.addEventListener("click", () => { if (q > 1) { q--; qVal.textContent = q; if (qInput) qInput.value = q; } });

  /* ---------- Collection sort ---------- */
  $("[data-sort]")?.addEventListener("change", (e) => {
    const u = new URL(location.href); u.searchParams.set("sort_by", e.target.value); location.href = u.toString();
  });
  $("#filtersToggle")?.addEventListener("click", () => $("#filters")?.classList.toggle("open"));
})();
