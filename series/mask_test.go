package series

import (
	"runtime"
	"slices"
	"testing"
)

func TestAsMask(t *testing.T) {
	nullable, err := NewNullable(
		[]bool{true, true, false, true},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	allNull, err := NewNullable([]bool{true, true}, []bool{false, false})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		values Series[bool]
		rows   []int
	}{
		{
			name:   "non-null",
			values: New([]bool{true, false, true, false}),
			rows:   []int{0, 2},
		},
		{
			name:   "nullable",
			values: nullable,
			rows:   []int{0, 3},
		},
		{
			name:   "all null",
			values: allNull,
			rows:   nil,
		},
		{
			name:   "empty non-null",
			values: Series[bool]{},
			rows:   nil,
		},
		{
			name:   "empty nullable",
			values: FromOptionals[bool](nil),
			rows:   nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			selection := AsMask(test.values)
			if selection.Len() != test.values.Len() {
				t.Fatalf("AsMask().Len() = %d, want %d", selection.Len(), test.values.Len())
			}
			if rows := slices.Collect(selection.Rows()); !slices.Equal(rows, test.rows) {
				t.Fatalf("AsMask().Rows() = %v, want %v", rows, test.rows)
			}
		})
	}
}

func BenchmarkAsMask(b *testing.B) {
	const length = 1 << 16
	physical := make([]bool, length)
	allPresent := make([]bool, length)
	partiallyNull := make([]bool, length)
	for i := range physical {
		physical[i] = i%2 == 0
		allPresent[i] = true
		partiallyNull[i] = i%4 != 0
	}
	allPresentValues, err := NewNullable(physical, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partiallyNullValues, err := NewNullable(physical, partiallyNull)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := NewNullable(physical, make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	benchmarks := []struct {
		name   string
		values Series[bool]
	}{
		{name: "non-null", values: New(physical)},
		{name: "nullable/all-present", values: allPresentValues},
		{name: "nullable/25%-null", values: partiallyNullValues},
		{name: "nullable/all-null", values: allNullValues},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var count int
			for b.Loop() {
				count = AsMask(benchmark.values).Count()
			}
			runtime.KeepAlive(count)
		})
	}
}
