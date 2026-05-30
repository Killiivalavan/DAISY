/* D.A.I.S.Y. v2 — Service Worker
 * Network-first for the UI shell so users always get fresh code.
 * Falls back to cache when offline. Static assets use stale-while-revalidate.
 */

const CACHE_VERSION = 'daisy-v2-shell-v2';
const SHELL_FILES = [
    '/',
    '/css/daisy.css',
    '/js/app.js',
    '/manifest.json',
    '/icons/daisy-192.png',
    '/icons/daisy-512.png',
];

// Files that must always come from the network (cache only as offline fallback)
const NETWORK_FIRST_FILES = ['/', '/css/daisy.css', '/js/app.js'];

self.addEventListener('install', function (event) {
    event.waitUntil(
        caches.open(CACHE_VERSION).then(function (cache) {
            return cache.addAll(SHELL_FILES);
        })
    );
    self.skipWaiting();
});

self.addEventListener('activate', function (event) {
    event.waitUntil(
        caches.keys().then(function (keys) {
            return Promise.all(
                keys.filter(function (k) { return k !== CACHE_VERSION; })
                    .map(function (k) { return caches.delete(k); })
            );
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', function (event) {
    if (event.request.method !== 'GET') return;

    var url = new URL(event.request.url);
    var path = url.pathname;

    // Network-first for the UI shell — always try to get fresh code
    if (NETWORK_FIRST_FILES.indexOf(path) !== -1 ||
        path.startsWith('/js/') || path.startsWith('/css/')) {

        event.respondWith(
            fetch(event.request).then(function (response) {
                // Update cache with the fresh response
                if (response.status === 200) {
                    var clone = response.clone();
                    caches.open(CACHE_VERSION).then(function (cache) {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            }).catch(function () {
                // Offline — fall back to cache
                return caches.match(event.request);
            })
        );
        return;
    }

    // Stale-while-revalidate for everything else (icons, manifest, etc.)
    event.respondWith(
        caches.match(event.request).then(function (cached) {
            var fetched = fetch(event.request).then(function (response) {
                if (response.status === 200) {
                    var clone = response.clone();
                    caches.open(CACHE_VERSION).then(function (cache) {
                        cache.put(event.request, clone);
                    });
                }
                return response;
            }).catch(function () {
                return cached || new Response('Offline', { status: 503 });
            });

            return cached || fetched;
        })
    );
});
