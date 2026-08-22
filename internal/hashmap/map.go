// Package hashmap provides the small generic hash table used by dataframe's
// custom-equivalence operations.
package hashmap

import "hash/maphash"

// Map is a hash table using a caller-supplied hash and equivalence relation.
// A Map must be created with New and is not safe for concurrent mutation.
type Map[K, V any] struct {
	hasher  maphash.Hasher[K]
	hash    maphash.Hash
	buckets map[uint64]int
	entries []entry[K, V]
}

type entry[K, V any] struct {
	key   K
	value V
	next  int
}

// New returns an empty Map using hasher. It panics when hasher is nil.
func New[K, V any](hasher maphash.Hasher[K]) *Map[K, V] {
	if hasher == nil {
		panic("hashmap: nil hasher")
	}
	m := &Map[K, V]{
		hasher:  hasher,
		buckets: make(map[uint64]int),
	}
	m.hash.SetSeed(maphash.MakeSeed())
	return m
}

// Get returns the value equivalent to key and whether it exists.
func (m *Map[K, V]) Get(key K) (V, bool) {
	for index := m.buckets[m.sum(key)] - 1; index >= 0; index = m.entries[index].next {
		entry := m.entries[index]
		if m.hasher.Equal(entry.key, key) {
			return entry.value, true
		}
	}
	var zero V
	return zero, false
}

// Set associates key with value, replacing an equivalent key's value.
func (m *Map[K, V]) Set(key K, value V) {
	hash := m.sum(key)
	for index := m.buckets[hash] - 1; index >= 0; index = m.entries[index].next {
		if m.hasher.Equal(m.entries[index].key, key) {
			m.entries[index].value = value
			return
		}
	}
	m.entries = append(m.entries, entry[K, V]{key: key, value: value, next: m.buckets[hash] - 1})
	m.buckets[hash] = len(m.entries)
}

// LoadOrStore returns the existing value for an equivalent key when present.
// Otherwise it stores and returns value. The loaded result reports which case
// occurred.
func (m *Map[K, V]) LoadOrStore(key K, value V) (actual V, loaded bool) {
	hash := m.sum(key)
	for index := m.buckets[hash] - 1; index >= 0; index = m.entries[index].next {
		entry := m.entries[index]
		if m.hasher.Equal(entry.key, key) {
			return entry.value, true
		}
	}
	m.entries = append(m.entries, entry[K, V]{key: key, value: value, next: m.buckets[hash] - 1})
	m.buckets[hash] = len(m.entries)
	return value, false
}

func (m *Map[K, V]) sum(key K) uint64 {
	m.hash.Reset()
	m.hasher.Hash(&m.hash, key)
	return m.hash.Sum64()
}
