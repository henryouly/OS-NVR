package openvino

import (
	"sync"
)

var cache = struct {
	previewCache *previewCache
}{}

func init() {
	cache.previewCache = newPreviewCache()
}

type previewCache struct {
	monitors map[string][]byte
	mu       *sync.Mutex
}

func newPreviewCache() *previewCache {
	return &previewCache{
		monitors: make(map[string][]byte),
		mu:       &sync.Mutex{},
	}
}

func (cache *previewCache) Set(monitorID string, buf []byte) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	cache.monitors[monitorID] = buf
}
