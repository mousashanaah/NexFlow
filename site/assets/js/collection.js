/* Collection page — product grid, sort, filters toggle, load more */
(() => {
  "use strict";
  const data = [
    { brand: "TaylorMade", name: "Qi10 Driver", img: "driver.svg", now: 1699, was: 2199, save: 23, rating: 5, reviews: 24, tag: "new" },
    { brand: "Callaway", name: "Paradym Ai Smoke Driver", img: "driver.svg", now: 1599, was: 2199, save: 27, rating: 5, reviews: 18 },
    { brand: "PING", name: "G430 MAX Driver", img: "driver.svg", now: 1749, was: 2099, save: 17, rating: 5, reviews: 32 },
    { brand: "Cobra", name: "Aerojet Driver", img: "driver.svg", now: 1299, was: 1899, save: 32, rating: 4, reviews: 16 },
    { brand: "TaylorMade", name: "Stealth 2 Driver", img: "driver.svg", now: 1199, was: 2199, save: 45, rating: 5, reviews: 64, tag: "deal" },
    { brand: "Titleist", name: "TSR2 Driver", img: "driver.svg", now: 1849, was: 2299, save: 20, rating: 5, reviews: 21 },
    { brand: "Callaway", name: "Rogue ST Max Driver", img: "driver.svg", now: 999, was: 1799, save: 44, rating: 4, reviews: 51 },
    { brand: "PING", name: "G425 Max Driver", img: "driver.svg", now: 1099, was: 1699, save: 35, rating: 5, reviews: 88 },
    { brand: "Cobra", name: "LTDx Driver", img: "driver.svg", now: 949, was: 1599, save: 41, rating: 4, reviews: 29 },
    { brand: "TaylorMade", name: "SIM2 Driver", img: "driver.svg", now: 899, was: 1699, save: 47, rating: 5, reviews: 73, tag: "deal" },
    { brand: "Titleist", name: "TSi3 Driver", img: "driver.svg", now: 1199, was: 1999, save: 40, rating: 5, reviews: 34 },
    { brand: "Callaway", name: "Epic Speed Driver", img: "driver.svg", now: 849, was: 1599, save: 47, rating: 4, reviews: 42 },
  ];

  const star = (n) => "★★★★★".slice(0, n) + "☆☆☆☆☆".slice(0, 5 - n);
  const grid = document.getElementById("colGrid");
  let shown = 9;

  const tag = (p) =>
    p.tag === "new" ? `<span class="tag new">New In</span>`
    : p.tag === "deal" ? `<span class="tag save">🔥 Deal</span>`
    : `<span class="tag save">Save ${p.save}%</span>`;

  const card = (p) => `
    <article class="card">
      <a href="product.html" class="card-media">
        ${tag(p)}
        <button class="wish" aria-label="Wishlist"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21s-7-4.5-9.5-8.5C.5 9 2 5.5 5.2 5.5c1.9 0 3.1 1 3.8 2 .7-1 1.9-2 3.8-2 3.2 0 4.7 3.5 2.7 7C19 16.5 12 21 12 21z"/></svg></button>
        <img src="assets/img/${p.img}" alt="${p.name}" loading="lazy" />
      </a>
      <div class="card-body">
        <span class="card-brand">${p.brand}</span>
        <a href="product.html"><h3 class="card-name">${p.name}</h3></a>
        <div class="stars">${star(p.rating)} <span>(${p.reviews})</span></div>
        <div class="price-row">
          <span class="now">AED ${p.now.toLocaleString()}</span>
          <span class="was">AED ${p.was.toLocaleString()}</span>
        </div>
        <button class="btn btn-dark add" data-add="${p.name}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 7h14l-1.5 12.5a1.5 1.5 0 0 1-1.5 1.5H8a1.5 1.5 0 0 1-1.5-1.5z"/><path d="M9 7V5a3 3 0 0 1 6 0v2"/></svg>
          Add to Bag
        </button>
      </div>
    </article>`;

  let current = [...data];
  const render = () => {
    grid.innerHTML = current.slice(0, shown).map(card).join("");
    document.getElementById("resultCount").textContent = `${current.length} products`;
    document.getElementById("loadMore").style.display = shown >= current.length ? "none" : "";
  };

  document.getElementById("sort").addEventListener("change", (e) => {
    const v = e.target.value;
    current = [...data];
    if (v === "low") current.sort((a, b) => a.now - b.now);
    else if (v === "high") current.sort((a, b) => b.now - a.now);
    else if (v === "save") current.sort((a, b) => b.save - a.save);
    else if (v === "rating") current.sort((a, b) => b.rating - a.rating || b.reviews - a.reviews);
    render();
  });

  document.getElementById("loadMore").addEventListener("click", () => { shown += 6; render(); });
  document.getElementById("filtersToggle")?.addEventListener("click", () => document.getElementById("filters").classList.toggle("open"));

  render();
})();
