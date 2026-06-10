/* Product page — gallery swap, option select, qty, related rail */
(() => {
  "use strict";

  /* Gallery */
  const mainImg = document.getElementById("mainImg");
  document.getElementById("thumbs")?.addEventListener("click", (e) => {
    const t = e.target.closest(".thumb");
    if (!t) return;
    document.querySelectorAll(".thumb").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mainImg.src = "assets/img/" + t.dataset.img;
  });

  /* Option groups */
  document.querySelectorAll(".opts").forEach((group) => {
    group.addEventListener("click", (e) => {
      const o = e.target.closest(".opt");
      if (!o) return;
      group.querySelectorAll(".opt").forEach((x) => x.classList.remove("active"));
      o.classList.add("active");
    });
  });

  /* Quantity */
  let q = 1;
  const qVal = document.getElementById("qVal");
  document.getElementById("qPlus")?.addEventListener("click", () => { q++; qVal.textContent = q; });
  document.getElementById("qMinus")?.addEventListener("click", () => { if (q > 1) { q--; qVal.textContent = q; } });

  /* Related rail */
  const related = [
    { brand: "Callaway", name: "Paradym Ai Smoke Driver", img: "driver.svg", now: 1599, was: 2199, rating: 5, reviews: 18, tagText: "Save 27%", tag: "save" },
    { brand: "PING", name: "G430 MAX Driver", img: "driver.svg", now: 1749, was: 2099, rating: 5, reviews: 32, tagText: "Save 17%", tag: "save" },
    { brand: "Titleist", name: "Pro V1 Balls (Dozen)", img: "golfball.svg", now: 189, was: 269, rating: 5, reviews: 142, tagText: "Save 30%", tag: "save" },
    { brand: "PING", name: "Hoofer Stand Bag", img: "bag.svg", now: 649, was: 899, rating: 5, reviews: 44, tagText: "Save 28%", tag: "save" },
  ];
  const star = (n) => "★★★★★".slice(0, n) + "☆☆☆☆☆".slice(0, 5 - n);
  const grid = document.getElementById("relatedGrid");
  if (grid) {
    grid.innerHTML = related.map((p) => `
      <article class="card">
        <a href="product.html" class="card-media">
          <span class="tag ${p.tag}">${p.tagText}</span>
          <button class="wish" aria-label="Wishlist"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21s-7-4.5-9.5-8.5C.5 9 2 5.5 5.2 5.5c1.9 0 3.1 1 3.8 2 .7-1 1.9-2 3.8-2 3.2 0 4.7 3.5 2.7 7C19 16.5 12 21 12 21z"/></svg></button>
          <img src="assets/img/${p.img}" alt="${p.name}" loading="lazy" />
        </a>
        <div class="card-body">
          <span class="card-brand">${p.brand}</span>
          <a href="product.html"><h3 class="card-name">${p.name}</h3></a>
          <div class="stars">${star(p.rating)} <span>(${p.reviews})</span></div>
          <div class="price-row"><span class="now">AED ${p.now.toLocaleString()}</span><span class="was">AED ${p.was.toLocaleString()}</span></div>
          <button class="btn btn-dark add" data-add="${p.name}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 7h14l-1.5 12.5a1.5 1.5 0 0 1-1.5 1.5H8a1.5 1.5 0 0 1-1.5-1.5z"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/></svg>
            Add to Bag
          </button>
        </div>
      </article>`).join("");
  }
})();
