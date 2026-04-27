/**
 * OllamaGate Service Worker
 * - Caches the shell (login + chat pages, CSS, JS)
 * - Serves cached shell while offline
 * - Never caches API calls (always network-first for those)
 */

const CACHE_NAME    = 'ollamagate-v1';
const OFFLINE_URL   = '/';

// Assets to pre-cache on install
const PRECACHE_URLS = [
  '/',
  '/static/js/app.js',
  '/static/manifest.json',
  'https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js',
];

// ─────────────────────────────────────────────
//  Install — pre-cache shell
// ─────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Pre-caching shell assets');
      // Use addAll but swallow individual failures gracefully
      return Promise.allSettled(
        PRECACHE_URLS.map(url => cache.add(url).catch(e => console.warn('[SW] Could not cache:', url, e)))
      );
    }).then(() => self.skipWaiting())
  );
});

// ─────────────────────────────────────────────
//  Activate — clean old caches
// ─────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => {
          console.log('[SW] Deleting old cache:', k);
          return caches.delete(k);
        })
      )
    ).then(() => self.clients.claim())
  );
});

// ─────────────────────────────────────────────
//  Fetch strategy
// ─────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // ── Always use network for API calls (streaming, auth, etc.)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(request));
    return;
  }

  // ── For navigation requests: network-first, fall back to cache
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then(resp => {
          // Cache a fresh copy
          const clone = resp.clone();
          caches.open(CACHE_NAME).then(c => c.put(request, clone));
          return resp;
        })
        .catch(() => {
          // Offline: serve cached page
          return caches.match(request)
            .then(cached => cached || caches.match(OFFLINE_URL));
        })
    );
    return;
  }

  // ── Static assets: cache-first with network fallback
  event.respondWith(
    caches.match(request).then(cached => {
      if (cached) return cached;
      return fetch(request).then(resp => {
        if (resp && resp.status === 200 && resp.type !== 'opaque') {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then(c => c.put(request, clone));
        }
        return resp;
      });
    })
  );
});

// ─────────────────────────────────────────────
//  Background sync placeholder (future use)
// ─────────────────────────────────────────────
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});
