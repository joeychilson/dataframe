package hashmap

import (
	"hash/maphash"
	"strings"
	"testing"
)

func TestNew_PanicsForNilHasherOrNegativeCapacity(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{name: "nil hasher is rejected", call: func() { New[string, int](nil, 0) }},
		{name: "negative capacity is rejected", call: func() { New[string, int](foldedStringHasher{}, -1) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("New did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestMap_UsesCustomEquivalence(t *testing.T) {
	values := New[string, int](foldedStringHasher{}, 2)
	values.Set("Go", 1)
	values.Set("Rust", 2)

	if value, ok := values.Get("go"); value != 1 || !ok {
		t.Fatalf("Get(\"go\") = (%d, %t), want (1, true)", value, ok)
	}
	if value, ok := values.Get("missing"); value != 0 || ok {
		t.Fatalf("Get(\"missing\") = (%d, %t), want (0, false)", value, ok)
	}

	values.Set("GO", 3)
	if value, _ := values.Get("go"); value != 3 {
		t.Fatalf("updated value = %d, want 3", value)
	}

	if value, loaded := values.LoadOrStore("gO", 4); value != 3 || !loaded {
		t.Fatalf("LoadOrStore(existing) = (%d, %t), want (3, true)", value, loaded)
	}
	if value, loaded := values.LoadOrStore("Zig", 5); value != 5 || loaded {
		t.Fatalf("LoadOrStore(new) = (%d, %t), want (5, false)", value, loaded)
	}
}

type foldedStringHasher struct{}

type largeKey [16]uint64

type largeValue [128]uint64

type collidingLargeHasher struct{}

func (foldedStringHasher) Hash(hash *maphash.Hash, _ string) {
	hash.WriteByte(0)
}

func (foldedStringHasher) Equal(left, right string) bool {
	return strings.EqualFold(left, right)
}

func (collidingLargeHasher) Hash(hash *maphash.Hash, _ largeKey) {
	hash.WriteByte(0)
}

func (collidingLargeHasher) Equal(left, right largeKey) bool {
	return left == right
}

func BenchmarkLargeCollidingEntries(b *testing.B) {
	const size = 64
	keys := make([]largeKey, size)
	stored := largeValue{1}
	values := New[largeKey, largeValue](collidingLargeHasher{}, size)
	for i := range keys {
		keys[i][0] = uint64(i + 1)
		values.Set(keys[i], stored)
	}

	b.Run("GetMiss", func(b *testing.B) {
		b.ReportAllocs()
		var result largeValue
		var found bool
		for b.Loop() {
			result, found = values.Get(largeKey{})
		}
		if found || result != (largeValue{}) {
			b.Fatal("Get found missing key")
		}
	})

	b.Run("LoadOrStoreHit", func(b *testing.B) {
		b.ReportAllocs()
		var result largeValue
		var loaded bool
		for b.Loop() {
			result, loaded = values.LoadOrStore(keys[0], largeValue{})
		}
		if !loaded || result != stored {
			b.Fatal("LoadOrStore did not load existing value")
		}
	})
}
