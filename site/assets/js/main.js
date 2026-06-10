/* =================================================================
   GOLF4LESS — interactions
   ================================================================= */
(() => {
  "use strict";

  /* ---------- Product data ---------- */
  const products = [
    { brand: "Titleist", name: "Pro V1 Golf Balls (Dozen)", img: "golfball.svg", now: 189, was: 269, tag: "save", tagText: "Save 30%", rating: 5, reviews: 142 },
    { brand: "Callaway", name: "Rogue ST Driver", img: "driver.svg", now: 1149, was: 1899, tag: "save", tagText: "Save 40%", rating: 4, reviews: 64 },
    { brand: "FootJoy", name: "WeatherSof Glove", img: "glove.svg", now: 39, was: 69, tag: "save", tagText: "Save 43%", rating: 5, reviews: 210 },
    { brand: "Under Armour", name: "Tour Tech Polo", img: "polo.svg", now: 159, was: 249, tag: "new", tagText: "New In", rating: 4, reviews: 38 },
    { brand: "Cobra", name: "Aerojet Fairway Wood", img: "driver.svg", now: 749, was: 1099, tag: "save", tagText: "Save 32%", rating: 5, reviews: 51 },
    { brand: "Golf4Less", name: "Custom Tour Balls (3 Dozen)", img: "golfball.svg", now: 219, was: 219, tag: "new", tagText: "Free Print", rating: 5, reviews: 96 },
    { brand: "PING", name: "Hoofer Stand Bag", img: "bag.svg", now: 649, was: 899, tag: "save", tagText: "Save 28%", rating: 5, reviews: 44 },
    { brand: "Srixon", name: "Soft Feel Balls (Dozen)", img: "golfball.svg", now: 89, was: 139, tag: "save", tagText: "Save 36%", rating: 4, reviews: 173 },
  ];

  const star = (n) => "★★★★★".slice(0, n) + "☆☆☆☆☆".slice(0, 5 - n);

  const grid = document.getElementById("productGrid");
  if (grid) {
    grid.innerHTML = products.map((p) => `
      <article class="card">
        <div class="card-media">
          <span class="tag ${p.tag}">${p.tagText}</span>
          <button class="wish" aria-label="Add to wishlist">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21s-7-4.5-9.5-8.5C.5 9 2 5.5 5.2 5.5c1.9 0 3.1 1 3.8 2 .7-1 1.9-2 3.8-2 3.2 0 4.7 3.5 2.7 7C19 16.5 12 21 12 21z"/></svg>
          </button>
          <img src="assets/img/${p.img}" alt="${p.name}" loading="lazy" />
        </div>
        <div class="card-body">
          <span class="card-brand">${p.brand}</span>
          <h3 class="card-name">${p.name}</h3>
          <div class="stars">${star(p.rating)} <span>(${p.reviews})</span></div>
          <div class="price-row">
            <span class="now">AED ${p.now.toLocaleString()}</span>
            ${p.was > p.now ? `<span class="was">AED ${p.was.toLocaleString()}</span>` : ""}
          </div>
          <button class="btn btn-dark add" data-add="${p.name}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 7h14l-1.5 12.5a1.5 1.5 0 0 1-1.5 1.5H8a1.5 1.5 0 0 1-1.5-1.5z"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/></svg>
            Add to Bag
          </button>
        </div>
      </article>`).join("");
  }

  /* ---------- Cart ---------- */
  let cart = 0;
  const cartCount = document.getElementById("cartCount");
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-add]");
    if (!btn) return;
    e.preventDefault();
    cart++;
    cartCount.textContent = cart;
    cartCount.animate(
      [{ transform: "scale(1)" }, { transform: "scale(1.5)" }, { transform: "scale(1)" }],
      { duration: 300, easing: "ease-out" }
    );
    const original = btn.innerHTML;
    btn.innerHTML = "✓ Added";
    btn.style.background = "var(--green-deep)";
    setTimeout(() => { btn.innerHTML = original; btn.style.background = ""; }, 1100);
  });

  /* ---------- Countdown (resets daily at midnight) ---------- */
  const cdH = document.getElementById("cd-h"), cdM = document.getElementById("cd-m"), cdS = document.getElementById("cd-s");
  if (cdH) {
    const tick = () => {
      const now = new Date();
      const end = new Date(now); end.setHours(24, 0, 0, 0);
      let diff = Math.floor((end - now) / 1000);
      const h = String(Math.floor(diff / 3600)).padStart(2, "0");
      const m = String(Math.floor((diff % 3600) / 60)).padStart(2, "0");
      const s = String(diff % 60).padStart(2, "0");
      cdH.textContent = h; cdM.textContent = m; cdS.textContent = s;
    };
    tick(); setInterval(tick, 1000);
  }

  /* ---------- Ball customiser ---------- */
  const ballInput = document.getElementById("ballInput");
  const ballText = document.getElementById("ballText");
  if (ballInput) {
    ballInput.addEventListener("input", () => {
      ballText.textContent = (ballInput.value.trim() || "YOUR TEXT").toUpperCase();
    });
    document.getElementById("swatches").addEventListener("click", (e) => {
      const sw = e.target.closest(".swatch");
      if (!sw) return;
      document.querySelectorAll(".swatch").forEach((s) => s.classList.remove("active"));
      sw.classList.add("active");
      ballText.style.color = sw.dataset.color;
    });
  }

  /* ---------- Header shadow on scroll ---------- */
  const header = document.getElementById("header");
  const onScroll = () => header.classList.toggle("scrolled", window.scrollY > 10);
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

  /* ---------- Reveal on scroll ---------- */
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
  }, { threshold: 0.14 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ---------- Animated counters ---------- */
  const counters = document.querySelectorAll("[data-count]");
  const cio = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target, target = +el.dataset.count;
      let cur = 0; const step = Math.max(1, target / 60);
      const run = () => { cur += step; if (cur >= target) { el.textContent = target.toLocaleString(); } else { el.textContent = Math.floor(cur).toLocaleString(); requestAnimationFrame(run); } };
      run(); cio.unobserve(el);
    });
  }, { threshold: 0.5 });
  counters.forEach((c) => cio.observe(c));

  /* ---------- Mobile drawer ---------- */
  const drawer = document.getElementById("drawer");
  document.getElementById("burger")?.addEventListener("click", () => drawer.classList.add("open"));
  drawer?.addEventListener("click", (e) => { if (e.target.matches("[data-close]")) drawer.classList.remove("open"); });

  /* ---------- Form niceties ---------- */
  const flash = (form, msg) => {
    const btn = form.querySelector('button[type="submit"], button:not([type])') || form.querySelector("button");
    if (!btn) return;
    const orig = btn.innerHTML; btn.innerHTML = msg; btn.style.background = "var(--green-deep)"; btn.style.color = "#fff";
    setTimeout(() => { btn.innerHTML = orig; btn.style.background = ""; btn.style.color = ""; form.reset(); }, 2000);
  };
  document.getElementById("newsForm")?.addEventListener("submit", (e) => { e.preventDefault(); flash(e.target, "✓ You're in!"); });
  document.getElementById("corpForm")?.addEventListener("submit", (e) => { e.preventDefault(); flash(e.target, "✓ Request sent!"); });
})();
