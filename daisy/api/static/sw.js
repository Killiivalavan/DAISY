/* D.A.I.S.Y. v2 — Service Worker
 * Caches the UI shell for offline display and fast repeat loads.
 */

const CACHE_NAME = 'daisy-v2-shell';
const SHELL_FILES = [
    '/',
    '/css/daisy.css',
    '/js/app.js',
    '/manifest.json',
    '/icons/daisy-192.png',
    '/icons/daisy-512.png',
];

self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function (cache) {
            return cache.addAll(SHELL_FILES);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', function (event) {
    event.waitUntil(
        caches.keys().then(function (keys) {
            return Promise.all(
                keys.filter(function (k) { return k !== CACHE_NAME; })
                    .map(function (k) { return caches.delete(k); })
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', function (event) {
    // Only cache GET requests to our own origin
    if (event.request.method !== 'GET') return;

    event.respondWith(
        caches.match(event.request).then(function (cached) {
            // Return cached response immediately if available
            if (cached) return cached;

            // Otherwise fetch from network
            return fetch(event.request).then(function (response) {
                // Don't cache API or WebSocket calls
                if (event.request.url.includes('/api/') ||
                    event.request.url.includes('/ws')) {
                    return response;
                }

                // Cache static assets for next time
                if (response.status === 200) {
                    var clone = response.clone();
                    caches.open(CACHE_NAME).then(function (cache) {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            }).catch(function () {
                // Offline and not cached — return nothing
                return new Response('Offline', { status: 503 });
            });
        })
    );
});
