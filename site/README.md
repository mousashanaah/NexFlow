# Golf4Less — Website

**Spend Less. Play More.** — MENA & Africa's digital-first value golf brand. An eGolf Megastore venture.

A fast, mobile-first, self-contained storefront marketing site. No build step, no dependencies — open `index.html` and it runs. Designed to translate directly into a Shopify theme.

## Structure
```
site/
├── index.html              # Full single-page storefront
├── assets/
│   ├── css/styles.css      # Design system + components
│   ├── js/main.js          # Cart, countdown, ball customiser, animations
│   └── img/                # Inline SVG brand + product art (logo, mascot, balls, gear)
```

## Brand
- **Colours:** Lime green `#7ac72e` / `#5fa81f`, ink black `#0e0e0e`, paper `#f6f7f4`
- **Type:** Anton (display) + Inter (body)
- **Voice:** Modern, energetic, fun, smart-value, community-focused

## Sections
Announcement marquee · Sticky header + mobile drawer · Hero with mascot · Brand strip · Daily Deal Drop (live countdown) · Shop-by-category grid · Product rail (add-to-bag) · **Live custom golf-ball printing preview** · Animated stats · Social/reels grid · Corporate bulk-quote form · Newsletter · Footer · WhatsApp float.

## Run locally
```bash
cd site && python3 -m http.server 8080
# open http://localhost:8080
```

## Porting to Shopify
- `index.html` sections map cleanly to Liquid sections/blocks.
- Product rail data (`assets/js/main.js`) is replaced by a `{% for product in collection.products %}` loop.
- Cart/checkout, customer accounts and payments are handled natively by Shopify.
- The custom-ball previewer can ship as a custom section/app block on the product page.
