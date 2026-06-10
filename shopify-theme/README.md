# Golf4Less — Shopify Theme

A custom **Online Store 2.0** theme for Golf4Less — *Spend Less. Play More.*
Digital-first value golf brand for MENA & Africa. An eGolf Megastore venture.

## Install
1. Download `golf4less-shopify-theme.zip` (in the repo root).
2. Shopify admin → **Online Store → Themes → Add theme → Upload zip file**.
3. Click **Customize** to open the theme editor, then **Publish** when ready.

> Built with JSON templates + sections, so every block is drag-and-drop editable in the Shopify theme editor — no code needed to rearrange the homepage.

## First-time setup
1. **Navigation** → create menus `main-menu` and `footer` (Online Store → Navigation). The header, sub-nav and footer columns read from these.
2. **Theme settings** (editor → Theme settings):
   - Brand colours (lime `#7ac72e` / ink `#0e0e0e`), logo, tagline.
   - WhatsApp number (enables the floating chat button) and social links.
3. **Homepage** (editor → Home): assign a collection to **Featured products** and a product to **Deal of the day**. Everything else has sensible defaults.
4. **Custom ball printing**: point the section's button at your custom-ball product or a contact page.
5. **Reviews stars**: optional — reads `product.metafields.reviews.rating` / `rating_count` (works with Shopify Product Reviews or Judge.me-style metafields).

## What's included
| Area | Sections / templates |
|---|---|
| **Home** | hero · brand-strip · deal-of-day (live countdown) · category-grid · featured-products · custom-balls (live previewer) · stats · social-reels · corporate · newsletter |
| **Catalog** | `main-collection` (Shopify filters + sort + pagination), `main-product` (gallery, variant options, qty, AJAX add-to-bag, dynamic checkout, related), `main-cart` (AJAX count, secure checkout) |
| **System** | search, list-collections, page, blog, article, 404, password, gift card, full customer account set |
| **Global** | header group (announcement + header + sub-nav), footer group, floating WhatsApp, mobile drawer |

## Structure
```
layout/theme.liquid
templates/*.json + *.liquid (+ customers/*)
sections/*.liquid  config/*.json  snippets/product-card.liquid
locales/en.default.json  assets/theme.css  assets/theme.js  assets/*.svg
```

## Notes
- Cart, checkout, payments, taxes and shipping are handled natively by Shopify.
- SVG art (logo, mascot, products) ships as placeholders — replace with real assets in `assets/` or via the editor image pickers.
- Checkout itself uses Shopify's secure hosted checkout; the in-repo `checkout.html` static mock was design reference only.
