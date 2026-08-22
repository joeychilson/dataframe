package hashmap

import (
	"hash/maphash"
	"strings"
	"testing"
)

type foldedStringHasher struct{}

func (foldedStringHasher) Hash(hash *maphash.Hash, _ string) {
	hash.WriteByte(0)
}

func (foldedStringHasher) Equal(left, right string) bool {
	return strings.EqualFold(left, right)
}

func TestMap(t *testing.T) {
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

func TestNewPanicsForNilHasher(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("New(nil) did not panic")
		}
	}()
	New[string, int](nil, 0)
}
